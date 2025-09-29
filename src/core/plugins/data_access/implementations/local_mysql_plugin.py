# uv run python -m src.plugins.implementations.mysql_plugin
# NOTE: This plugin requires the 'mysql-connector-python' package.
# You can install it using: pip install mysql-connector-python

import os
import time
from typing import Any, List, Tuple, Dict

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from mysql.connector import errorcode

from src.core.plugins.data_access.data_access_plugin_core import DBConnector, PluginManager


class MySQLConnectionManager:
    """
    A singleton class for managing MySQL server connections.
    
    This manager handles connection setup and maintenance using credentials 
    configured via environment variables.
    
    Required Environment Variables:
        MYSQL_USER: Username for MySQL authentication
        MYSQL_PASSWORD: Password for MySQL authentication
        MYSQL_DATABASE: Target database name
        MYSQL_HOST: (Optional) Server hostname (default: localhost)
        MYSQL_PORT: (Optional) Server port (default: 3306)
    """
    load_dotenv()
    _instance = None
    _pool: Dict[str, mysql.connector.MySQLConnection] = {}

    def __new__(cls):
        """
        Create a new instance of the singleton, or return the existing one.
        """
        if cls._instance is None:
            cls._instance = super(MySQLConnectionManager, cls).__new__(cls)
        return cls._instance

    def get_connection(self, cfg: Dict[str, Any] | None = None) -> mysql.connector.MySQLConnection:
        """
        Get a MySQL connection for the given configuration, establishing it if it doesn't exist or is closed.

        Returns:
            mysql.connector.MySQLConnection: The active MySQL connection object for the config key.
        """
        key = self._make_key(cfg)
        if key not in self._pool or not self._pool[key].is_connected():
            self._pool[key] = self._establish_mysql_connection(cfg)
        return self._pool[key]

    def _make_key(self, cfg: Dict[str, Any] | None) -> str:
        import hashlib, json
        payload = json.dumps(cfg or {}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _establish_mysql_connection(self, cfg: Dict[str, Any] | None = None) -> mysql.connector.MySQLConnection:
        """
        Establish a connection to MySQL using environment variables.

        Returns:
            mysql.connector.MySQLConnection: A new MySQL connection object.
            
        Raises:
            ValueError: If required environment variables are missing.
            ConnectionError: If the connection fails for any reason.
        """
        config = {
            'host': (cfg or {}).get('host') or os.environ.get('MYSQL_HOST', 'localhost'),
            'user': (cfg or {}).get('user') or os.environ.get('MYSQL_USER'),
            'password': (cfg or {}).get('password') or os.environ.get('MYSQL_PASSWORD'),
            'database': (cfg or {}).get('database') or os.environ.get('MYSQL_DATABASE'),
            'port': int(((cfg or {}).get('port') or os.environ.get('MYSQL_PORT', '3306')))
        }

        missing_vars = [key.upper() for key, value in config.items() if value is None and key in ['user', 'password', 'database']]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        try:
            connection = mysql.connector.connect(**config)
            return connection
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                raise ConnectionError("Authentication failed: Invalid user name or password.")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                raise ConnectionError(f"Database '{config['database']}' does not exist.")
            else:
                raise ConnectionError(f"Failed to establish MySQL connection: {err}") from err


class MySQLConnector(DBConnector):
    """
    MySQLConnector is a plugin that interacts with a MySQL database.
    It implements the common DBConnector interface with connection state management.
    """
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, **kwargs: Any):
        """
        Initializes the MySQLConnector.
        
        Args:
            max_retries (int): Maximum number of retry attempts for queries.
            retry_delay (float): Delay in seconds between retries.
        """
        self.connection = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connected = False
        # Persist constructor kwargs to participate in per-tenant connection pooling
        # and allow overrides of env-based config.
        self._cfg: Dict[str, Any] | None = dict(kwargs) if kwargs else None

    @property
    def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to the database.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self._connected and self.connection is not None and self.connection.is_connected()

    def connect(self) -> Tuple[mysql.connector.MySQLConnection, str]:
        """
        Establish the connection using the MySQLConnectionManager singleton.
        
        Returns:
            tuple: A tuple containing the connection object and the database name.
            
        Raises:
            ConnectionError: If the connection cannot be established.
        """
        try:
            cfg = getattr(self, "_cfg", None)
            self.connection = MySQLConnectionManager().get_connection(cfg)
            self._connected = True
            return self.connection, self.connection.database
        except Exception as e:
            self._connected = False
            print(f"Connection failed: {type(e).__name__}: {str(e)}")
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}") from e

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a query against MySQL with retry logic and return a DataFrame.
        
        Args:
            query (str): A SQL query string.
            
        Returns:
            pd.DataFrame: Query results as a pandas DataFrame (empty for non-SELECT).
            
        Raises:
            ConnectionError: If the query fails after all retry attempts.
        """
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            try:
                if not self.is_connected:
                    self.connect()
                
                cursor = self.connection.cursor()
                cursor.execute(query)

                if cursor.description is None:  # For INSERT, UPDATE, DELETE
                    self.connection.commit()
                    cursor.close()
                    return pd.DataFrame()

                rows = cursor.fetchall()
                description = cursor.description
                cursor.close()
                columns = [desc[0] for desc in description]
                df = pd.DataFrame(rows, columns=columns)
                return df

            except mysql.connector.Error as err:
                last_error = err
                attempts += 1
                self._connected = False 
                print(f"Attempt {attempts} failed: {type(err).__name__}: {str(err)}")
                
                if err.errno in (errorcode.ER_ACCESS_DENIED_ERROR, errorcode.ER_BAD_DB_ERROR):
                    raise ConnectionError(f"Query failed with a non-recoverable error: {str(last_error)}") from last_error

                if attempts < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
            except Exception as e:
                last_error = e
                attempts += 1
                self._connected = False
                print(f"Attempt {attempts} failed: {type(e).__name__}: {str(e)}")
                if attempts < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

        raise ConnectionError(f"Query failed after {self.max_retries} attempts. Last error: {str(last_error)}")

    def disconnect(self) -> None:
        """
        Close the database connection and clean up resources.
        """
        # Only clear local references; underlying pooled connection lives in the manager
        self.connection = None
        self._connected = False


def dataframe_from_response(response: Tuple[List[Tuple], Any]) -> pd.DataFrame:
    """
    Convert a MySQL response tuple into a pandas DataFrame.
    
    Args:
        response (tuple): A tuple containing a list of rows and the cursor description.
        
    Returns:
        pd.DataFrame: A DataFrame containing the query results.
    """
    rows, description = response
    if not rows or not description:
        return pd.DataFrame()
    
    columns = [desc[0] for desc in description]
    return pd.DataFrame(rows, columns=columns)



# uv run python -m src.plugins_data_access.implementations.local_mysql_plugin
if __name__ == "__main__":
    PluginManager.register_plugin("mysql", MySQLConnector)
    mysql_plugin = None
    try:
        mysql_plugin = PluginManager.get_plugin("mysql")
        mysql_plugin.connect()
        print("Successfully connected to MySQL.")
        
        query = """
        SELECT * FROM control_minutedata 
        WHERE controlId = '2916276' 
        AND locationId = '11688'
        AND timeStamp BETWEEN '2025-02-05' AND '2025-05-05'
        """
        df = mysql_plugin.execute_query(query)
        print("\nQuery result:")
        print(df)
        
    except (ValueError, ConnectionError) as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__}: {e}")
    finally:
        if mysql_plugin and mysql_plugin.is_connected:
            mysql_plugin.disconnect()
            print("\nDisconnected from MySQL.")
