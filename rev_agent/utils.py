from datetime import datetime
import pytz


def get_current_datetime_context() -> str:
    """
    Get current date and time context for the agent.
    Returns a formatted string with current date, time, and timezone.
    """
    try:
        # Get current UTC time
        utc_now = datetime.now(pytz.UTC)
        
        # Format the context string
        context = f"""**Current System Information:**
- Date: {utc_now.strftime('%A, %B %d, %Y')}
- Time: {utc_now.strftime('%I:%M %p UTC')}
- Timezone: UTC
- ISO Format: {utc_now.isoformat()}

Use this current date and time information when responding to date/time-related queries."""
        
        return context
    except Exception as e:
        # Fallback if timezone handling fails
        now = datetime.now()
        return f"""**Current System Information:**
- Date: {now.strftime('%A, %B %d, %Y')}
- Time: {now.strftime('%I:%M %p')}

Use this current date and time information when responding to date/time-related queries."""


def get_simple_date() -> str:
    """Get simple formatted current date"""
    return datetime.now().strftime('%A, %B %d, %Y')


def get_simple_time() -> str:
    """Get simple formatted current time"""
    return datetime.now().strftime('%I:%M %p')

