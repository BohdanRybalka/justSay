"""Insights API — the launch panel's personal figures."""

from fastapi import APIRouter

from app.transcripts import insights
from app.transcripts.store_errors import store_busy_as_503

router = APIRouter(prefix="/insights", tags=["Insights"])


@router.get("", response_model=insights.Insights)
async def get_insights(days: insights.ChartSpan = insights.ChartSpan.MONTH):
    with store_busy_as_503():
        return insights.compute_insights(days)
