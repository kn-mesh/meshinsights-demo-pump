# Plots
- Utilize Plotly for all plots

# General
- I always use dark mode
- When displaying raw dataframes always put in an st.expander


# What to watchout for
- When plots are generated for multiple devices, need to have distinct keys in the streamlit components to avoid errors.
- If there is interaction in the app leverage st.fragment to avoid re-rendering the entire app. st.fragment docs:
```
Decorator to turn a function into a fragment which can rerun independently of the full app.

When a user interacts with an input widget created inside a fragment, Streamlit only reruns the fragment instead of the full app. If run_every is set, Streamlit will also rerun the fragment at the specified interval while the session is active, even if the user is not interacting with your app.

To trigger an app rerun from inside a fragment, call st.rerun() directly. To trigger a fragment rerun from within itself, call st.rerun(scope="fragment"). Any values from the fragment that need to be accessed from the wider app should generally be stored in Session State.

When Streamlit element commands are called directly in a fragment, the elements are cleared and redrawn on each fragment rerun, just like all elements are redrawn on each app rerun. The rest of the app is persisted during a fragment rerun. When a fragment renders elements into externally created containers, the elements will not be cleared with each fragment rerun. Instead, elements will accumulate in those containers with each fragment rerun, until the next app rerun.

Calling st.sidebar in a fragment is not supported. To write elements to the sidebar with a fragment, call your fragment function inside a with st.sidebar context manager.

Fragment code can interact with Session State, imported modules, and other Streamlit elements created outside the fragment. Note that these interactions are additive across multiple fragment reruns. You are responsible for handling any side effects of that behavior.

Warning

Fragments can only contain widgets in their main body. Fragments can't render widgets to externally created containers.
Function signature[source]
st.fragment(func=None, *, run_every=None)
Parameters
func (callable)
The function to turn into a fragment.
run_every (int, float, timedelta, str, or None)
The time interval between automatic fragment reruns. This can be one of the following:

None (default).
An int or float specifying the interval in seconds.
A string specifying the time in a format supported by Pandas' Timedelta constructor, e.g. "1d", "1.5 days", or "1h23s".
A timedelta object from Python's built-in datetime library, e.g. timedelta(days=1).
If run_every is None, the fragment will only rerun from user-triggered events.
```



# When Using st.dataframe for selection 1 or more rows, here's the documentation:
```
Display a dataframe as an interactive table.

This command works with a wide variety of collection-like and dataframe-like object types.
Function signature[source]
st.dataframe(data=None, width="stretch", height="auto", *, use_container_width=None, hide_index=None, column_order=None, column_config=None, key=None, on_select="ignore", selection_mode="multi-row", row_height=None)
Parameters
data (dataframe-like, collection-like, or None)
The data to display.

Dataframe-like objects include dataframe and series objects from popular libraries like Dask, Modin, Numpy, pandas, Polars, PyArrow, Snowpark, Xarray, and more. You can use database cursors and clients that comply with the Python Database API Specification v2.0 (PEP 249). Additionally, you can use anything that supports the Python dataframe interchange protocol.

For example, you can use the following:

pandas.DataFrame, pandas.Series, pandas.Index, pandas.Styler, and pandas.Array
polars.DataFrame, polars.LazyFrame, and polars.Series
snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table
If a data type is not recognized, Streamlit will convert the object to a pandas.DataFrame or pyarrow.Table using a .to_pandas() or .to_arrow() method, respectively, if available.

If data is a pandas.Styler, it will be used to style its underlying pandas.DataFrame. Streamlit supports custom cell values, colors, and font weights. It does not support some of the more exotic styling options, like bar charts, hovering, and captions. For these styling options, use column configuration instead. Text and number formatting from column_config always takes precedence over text and number formatting from pandas.Styler.

Collection-like objects include all Python-native Collection types, such as dict, list, and set.

If data is None, Streamlit renders an empty table.
width ("stretch", "content", or int)
The width of the dataframe element. This can be one of the following:

"stretch" (default): The width of the element matches the width of the parent container.
"content": The width of the element matches the width of its content, but doesn't exceed the width of the parent container.
An integer specifying the width in pixels: The element has a fixed width. If the specified width is greater than the width of the parent container, the width of the element matches the width of the parent container.
height (int or "auto")
The height of the dataframe element. This can be one of the following:

"auto" (default): Streamlit sets the height to show at most ten rows.
An integer specifying the height in pixels: The element has a fixed height.
Vertical scrolling within the dataframe element is enabled when the height does not accommodate all rows.
use_container_width (bool)
delete
use_container_width is deprecated and will be removed in a future release. For use_container_width=True, use width="stretch".
Whether to override width with the width of the parent container. If this is True (default), Streamlit sets the width of the dataframe to match the width of the parent container. If this is False, Streamlit sets the dataframe's width according to width.
hide_index (bool or None)
Whether to hide the index column(s). If hide_index is None (default), the visibility of index columns is automatically determined based on the data.
column_order (Iterable[str] or None)
The ordered list of columns to display. If this is None (default), Streamlit displays all columns in the order inherited from the underlying data structure. If this is a list, the indicated columns will display in the order they appear within the list. Columns may be omitted or repeated within the list.

For example, column_order=("col2", "col1") will display "col2" first, followed by "col1", and will hide all other non-index columns.

column_order does not accept positional column indices and can't move the index column(s).
column_config (dict or None)
Configuration to customize how columns are displayed. If this is None (default), columns are styled based on the underlying data type of each column.

Column configuration can modify column names, visibility, type, width, format, and more. If this is a dictionary, the keys are column names (strings) and/or positional column indices (integers), and the values are one of the following:

None to hide the column.
A string to set the display label of the column.
One of the column types defined under st.column_config. For example, to show a column as dollar amounts, use st.column_config.NumberColumn("Dollar values", format="$ %d"). See more info on the available column types and config options here.
To configure the index column(s), use "_index" as the column name, or use a positional column index where 0 refers to the first index column.
key (str)
An optional string to use for giving this element a stable identity. If key is None (default), this element's identity will be determined based on the values of the other parameters.

Additionally, if selections are activated and key is provided, Streamlit will register the key in Session State to store the selection state. The selection state is read-only.
on_select ("ignore" or "rerun" or callable)
How the dataframe should respond to user selection events. This controls whether or not the dataframe behaves like an input widget. on_select can be one of the following:

"ignore" (default): Streamlit will not react to any selection events in the dataframe. The dataframe will not behave like an input widget.
"rerun": Streamlit will rerun the app when the user selects rows, columns, or cells in the dataframe. In this case, st.dataframe will return the selection data as a dictionary.
A callable: Streamlit will rerun the app and execute the callable as a callback function before the rest of the app. In this case, st.dataframe will return the selection data as a dictionary.
selection_mode ("single-row", "multi-row", "single-column", "multi-column", "single-cell", "multi-cell", or Iterable of these)
The types of selections Streamlit should allow when selections are enabled with on_select. This can be one of the following:

"multi-row" (default): Multiple rows can be selected at a time.
"single-row": Only one row can be selected at a time.
"multi-column": Multiple columns can be selected at a time.
"single-column": Only one column can be selected at a time.
"multi-cell": A rectangular range of cells can be selected.
"single-cell": Only one cell can be selected at a time.
An Iterable of the above options: The table will allow selection based on the modes specified. For example, to allow the user to select multiple rows and multiple cells, use ["multi-row", "multi-cell"].
When column selections are enabled, column sorting is disabled.
row_height (int or None)
The height of each row in the dataframe in pixels. If row_height is None (default), Streamlit will use a default row height, which fits one line of text.
Returns
(element or dict)
If on_select is "ignore" (default), this command returns an internal placeholder for the dataframe element that can be used with the .add_rows() method. Otherwise, this command returns a dictionary-like object that supports both key and attribute notation. The attributes are described by the DataframeState dictionary schema.


DataframeSelectionState:
The schema for the dataframe selection state.

The selection state is stored in a dictionary-like object that supports both key and attribute notation. Selection states cannot be programmatically changed or set through Session State.

Warning

If a user sorts a dataframe, row selections will be reset. If your users need to sort and filter the dataframe to make selections, direct them to use the search function in the dataframe toolbar instead.
Attributes
rows (list[int])
The selected rows, identified by their integer position. The integer positions match the original dataframe, even if the user sorts the dataframe in their browser. For a pandas.DataFrame, you can retrieve data from its integer position using methods like .iloc[] or .iat[].
columns (list[str])
The selected columns, identified by their names.
cells (list[tuple[int, str]])
The selected cells, provided as a tuple of row integer position and column name. For example, the first cell in a column named "col 1" is represented as (0, "col 1"). Cells within index columns are not returned.
```