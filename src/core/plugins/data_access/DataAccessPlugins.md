# Database Query Plugins

A lightweight plugin system for read-only queries across different cloud and IoT data sources. This library provides a consistent way to authenticate and query multiple databases while preserving their native query languages.

## Features

- Read-only query interface for multiple databases
- Simple plugin architecture for adding new data sources
- Maintains native query language support for each database
- Handles authentication and connection management
- Focus on querying data, not modifying it

## Currently Supported Databases

### Azure Data Explorer (ADX)
- Supports Service Principal authentication
- Full Kusto query language support
- Automatic retry logic for failed queries
- Converts results to pandas DataFrames

## Quick Start

```python
from db_plugins import PluginManager
from db_plugins.connectors import ADXConnector

# Register the ADX plugin
PluginManager.register_plugin("adx", ADXConnector)

# Get an authenticated connection (uses Service Principal)
adx = PluginManager.get_plugin("adx")

# Execute read-only queries using native Kusto
result = adx.execute_query("""
    DeviceTelemetry
    | where deviceId == 'device_123'
    | summarize avg(temperature) by bin(timestamp, 1h)
""")
```

## Authentication

### ADX Service Principal Setup
The ADX connector uses Azure Service Principal authentication. You'll need to set up these environment variables:

```
AZURE_TENANT_ID=your_tenant_id          # Azure AD tenant ID
AZURE_CLIENT_ID=your_client_id          # Service Principal client ID
AZURE_CLIENT_SECRET=your_client_secret  # Service Principal secret
KUSTO_CLUSTER=your_cluster_url         # ADX cluster URL
KUSTO_DATABASE=your_database           # Target database name
```

To set up a Service Principal for ADX:
1. Create a Service Principal in Azure AD
2. Assign appropriate read permissions to your ADX database
3. Get the credentials and configure environment variables

## Adding New Database Connectors

Create a new read-only connector by implementing the `DBConnector` interface:

```python
from db_plugins import DBConnector

class MyDatabaseConnector(DBConnector):
    def connect(self):
        # Implement read-only database connection
        pass

    def execute_query(self, query: str):
        # Execute read-only queries using database's native language
        pass

# Register your connector
PluginManager.register_plugin("my_database", MyDatabaseConnector)
```

