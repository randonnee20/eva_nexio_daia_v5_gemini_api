"""
DAIA - IP 기반 일일 사용량 제한
백엔드: Supabase PostgreSQL

환경변수:
  SUPABASE_URL   - 프로젝트 URL
  SUPABASE_KEY   - anon public key
  DAILY_LIMIT    - 하루 최대 횟수 (기본 5)

테이블 DDL: supabase_setup.sql 참고
"""
from __future__ import annotations
import os, hashlib
from datetime import date
from typing import Tuple

DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "5"))


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _hash_ip(ip: str) -> str:
    """IP를 단방향 해시 저장 (개인정보 보호)"""
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:32]


def _get_client():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def _extract_ip(request) -> str:
    """Gradio gr.Request에서 IP 추출, 실패 시 'unknown' 반환"""
    if request is None:
        return "unknown"
    try:
        # X-Forwarded-For (로드밸런서/프록시 환경)
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        # 직접 연결
        return str(getattr(request, "client", None) or "unknown")
    except Exception:
        return "unknown"


# ── public API ────────────────────────────────────────────────────────────────

def check_and_increment(request) -> Tuple[bool, int, int]:
    """
    사용 가능 여부 확인 + 카운트 증가 (원자적 처리).

    Returns
    -------
    allowed : bool
        True → 사용 가능 / False → 한도 초과
    current : int
        증가 후 오늘 사용 횟수
    limit : int
        하루 최대 허용 횟수
    """
    ip = _extract_ip(request)
    client = _get_client()

    # Supabase 미설정 → 제한 없음 (로컬 개발용)
    if client is None:
        return True, 0, DAILY_LIMIT

    today = str(date.today())
    ip_hash = _hash_ip(ip)

    try:
        res = (
            client.table("usage_log")
            .select("count")
            .eq("ip_hash", ip_hash)
            .eq("usage_date", today)
            .execute()
        )
        rows = res.data or []
        current = rows[0]["count"] if rows else 0

        if current >= DAILY_LIMIT:
            return False, current, DAILY_LIMIT

        # 카운트 upsert
        if rows:
            client.table("usage_log").update(
                {"count": current + 1}
            ).eq("ip_hash", ip_hash).eq("usage_date", today).execute()
        else:
            client.table("usage_log").insert(
                {"ip_hash": ip_hash, "usage_date": today, "count": 1}
            ).execute()

        return True, current + 1, DAILY_LIMIT

    except Exception as e:
        # DB 오류 시 허용 (서비스 중단 방지)
        import logging
        logging.getLogger("daia").warning(f"rate_limiter DB 오류 (허용 처리): {e}")
        return True, 0, DAILY_LIMIT


def get_remaining(request) -> int:
    """오늘 남은 사용 횟수 반환 (카운트 증가 없음)"""
    ip = _extract_ip(request)
    client = _get_client()
    if client is None:
        return DAILY_LIMIT

    today = str(date.today())
    ip_hash = _hash_ip(ip)
    try:
        res = (
            client.table("usage_log")
            .select("count")
            .eq("ip_hash", ip_hash)
            .eq("usage_date", today)
            .execute()
        )
        rows = res.data or []
        used = rows[0]["count"] if rows else 0
        return max(0, DAILY_LIMIT - used)
    except Exception:
        return DAILY_LIMIT
