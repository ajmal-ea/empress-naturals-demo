from dateutil.parser import isoparse
from datetime import datetime, timezone

def parseDate(val):
    if isinstance(val, str):
        val = isoparse(val)
        return val
    elif isinstance(val, datetime):
        return val
    else:
        raise ValueError(f"Invalid date type: {type(val)}")