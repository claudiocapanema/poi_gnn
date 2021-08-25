import datetime as dt
import pytz

class DatetimesUtils:

    @classmethod
    def date_from_str_to_datetime(date):
        pattern = '%Y-%m-%d %H:%M:%S'
        return dt.datetime.strptime(date, pattern)

    @classmethod
    def convert_tz(cls, datetime, from_tz, to_tz):
        datetime = datetime.replace(tzinfo=from_tz)
        datetime = datetime.astimezone(pytz.timezone(to_tz))
        return  datetime
    
    @classmethod
    def point_duration(cls, point_0, point_1):
        duration = point_1 - point_0
        duration = duration.days*24*60 + duration.seconds/60
        return round(duration)
