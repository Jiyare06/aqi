"""Re-export of the shared CPCB-style AQI function (shared/src/cpcb_aqi)."""

from cpcb_aqi import *  # noqa: F401,F403
from cpcb_aqi import __all__  # noqa: F401
from cpcb_aqi.aqi import _subindex  # noqa: F401  (used by tests/test_aqi_cpcb.py)
