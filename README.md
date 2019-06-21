# Paper trading tests of machine learning algorithms

This package implements several machine learning algorithms for stock prediction to evaluate performance out of sample.

the project is base on Zipline backtest platform, see the Zipline: http://www.zipline.io/
, Sklearn library: https://scikit-learn.org/
, and TA library: https://www.ta-lib.org/


### Step 1: Install External Library  ###
 
`pip install -r requirements.txt`


### Step 2: Install Package & Create Data Bundle  ###

`python setup.py install`


### Step 3: Create Data Bundles  ###
Zipline provides a bundle called csvdir, which allows users to ingest data from .csv files. The format of the files should be in OHLCV format, with dates, dividends, and splits. Once you have your data in the correct format, you can edit your `extension.py` file in `~/.zipline/extension.py` and import the csvdir bundle.

```
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

register(
    'custom-csvdir-bundle',# your bundle name
    csvdir_equities(
        ['daily'],#your subdirectory of the path below 
        '/path/to/your/csv',
    ),
    calendar_name='NYSE', # US equities
)

```
Note that the parameter ['daily'] is recognised in zipline to be daily data (for minute data this parameter can be changed to ['minute']
To finally ingest our data, we can run: `zipline ingest -b custom-csvdir-bundle`

More details about data bundles in Zipline: http://www.zipline.io/bundles.html#ingesting-data-from-csv-files


### Step 3: Run Backtest by Machine Learning algorithm  ###
`python toollib/algorithm/test.py`

### or Step 3: Run Backtest by NEAT algorithm  ###
`python toollib/algorithm/test_neat.py`
