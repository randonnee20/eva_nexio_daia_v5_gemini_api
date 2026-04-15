-- DAIA - Supabase 사용량 테이블 설정
-- Supabase 대시보드 → SQL Editor에 붙여넣고 실행

-- 사용량 로그 테이블
CREATE TABLE IF NOT EXISTS usage_log (
    id          BIGSERIAL PRIMARY KEY,
    ip_hash     TEXT        NOT NULL,          -- SHA-256 앞 32자 (개인정보 비저장)
    usage_date  DATE        NOT NULL DEFAULT CURRENT_DATE,
    count       INTEGER     NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (ip_hash, usage_date)               -- IP + 날짜 복합 유니크
);

-- updated_at 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_updated_at ON usage_log;
CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON usage_log
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- 날짜 인덱스 (오래된 데이터 조회 최적화)
CREATE INDEX IF NOT EXISTS idx_usage_log_date ON usage_log (usage_date);

-- Row Level Security: anon key로 읽기/쓰기만 허용
ALTER TABLE usage_log ENABLE ROW LEVEL SECURITY;

-- anon 역할에 SELECT / INSERT / UPDATE 허용
CREATE POLICY "anon_select" ON usage_log FOR SELECT USING (true);
CREATE POLICY "anon_insert" ON usage_log FOR INSERT WITH CHECK (true);
CREATE POLICY "anon_update" ON usage_log FOR UPDATE USING (true);

-- (선택) 30일 지난 데이터 자동 삭제 — pg_cron 필요
-- SELECT cron.schedule('cleanup-usage-log', '0 3 * * *',
--   $$DELETE FROM usage_log WHERE usage_date < CURRENT_DATE - INTERVAL '30 days'$$);
