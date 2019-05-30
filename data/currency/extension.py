import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities




register(
    'custom-currency-csvdir-bundle',
    csvdir_equities(
        ['minute'],
	'/home/sustechcs/proj/stock_trading/csv/currency',
    ),
	calendar_name='24/7', #AlwaysOpenCalendar

)
