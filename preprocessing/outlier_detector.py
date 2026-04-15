"""
EVA NEXIO DAIA - 이상치 탐지 및 처리
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
import yaml

from utils.logger import get_logger

logger = get_logger()


class OutlierDetector:
    """이상치 탐지 및 처리 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.outlier_config = self.config.get('preprocessing', {}).get('outlier', {})
        self.outlier_info = {}  # 이상치 정보 저장
        self.boundaries = {}  # 경계값 저장
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def detect_outliers(self, df: pd.DataFrame, method: str = None) -> Dict:
        """
        이상치 탐지
        
        Args:
            df: 데이터프레임
            method: 'iqr', 'zscore', 'isolation_forest'
        """
        if method is None:
            method = self.outlier_config.get('method', 'iqr')
        
        logger.log_step(f"이상치 탐지 ({method})", "START")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}
        
        with logger.log_timer("이상치 탐지"):
            for col in numeric_cols:
                if method == 'iqr':
                    outliers, bounds = self._detect_iqr(df[col])
                elif method == 'zscore':
                    outliers, bounds = self._detect_zscore(df[col])
                elif method == 'isolation_forest':
                    outliers, bounds = self._detect_isolation_forest(df[[col]])
                else:
                    logger.warning(f"알 수 없는 방법: {method}, IQR 사용")
                    outliers, bounds = self._detect_iqr(df[col])
                
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    outlier_info[col] = {
                        'count': int(outlier_count),
                        'ratio': float(outlier_count / len(df)),
                        'indices': df[outliers].index.tolist(),
                        'values': df.loc[outliers, col].tolist(),
                        'bounds': bounds
                    }
                    self.boundaries[col] = bounds
        
        self.outlier_info = outlier_info
        total_outliers = sum(info['count'] for info in outlier_info.values())
        logger.info(f"✅ 이상치 탐지: {len(outlier_info)}개 컬럼, 총 {total_outliers}개")
        
        return outlier_info
    
    def _detect_iqr(self, series: pd.Series) -> Tuple[pd.Series, Dict]:
        """IQR 방법"""
        multiplier = self.outlier_config.get('iqr_multiplier', 1.5)
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (series < lower_bound) | (series > upper_bound)
        bounds = {'lower': float(lower_bound), 'upper': float(upper_bound)}
        
        return outliers, bounds
    
    def _detect_zscore(self, series: pd.Series) -> Tuple[pd.Series, Dict]:
        """Z-score 방법"""
        threshold = self.outlier_config.get('zscore_threshold', 3.0)
        
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series([False] * len(series), index=series.index), {'mean': mean, 'std': 0}
        
        z_scores = np.abs((series - mean) / std)
        outliers = z_scores > threshold
        
        bounds = {
            'mean': float(mean),
            'std': float(std),
            'threshold': threshold
        }
        
        return outliers, bounds
    
    def _detect_isolation_forest(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Isolation Forest 방법"""
        contamination = self.outlier_config.get('contamination', 0.1)
        
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(df)
        
        outliers = pd.Series(predictions == -1, index=df.index)
        bounds = {'method': 'isolation_forest', 'contamination': contamination}
        
        return outliers, bounds
    
    def handle_outliers(self, df: pd.DataFrame, strategy: Dict = None,
                       fit: bool = True) -> pd.DataFrame:
        """
        이상치 처리
        
        Args:
            df: 데이터프레임
            strategy: 컬럼별 처리 전략 {'col': 'clip', 'remove', 'cap', 'fill'}
            fit: True면 경계값 학습
        """
        logger.log_step("이상치 처리", "START")
        df_copy = df.copy()
        
        # 먼저 이상치 탐지
        if fit:
            outlier_info = self.detect_outliers(df_copy)
        else:
            outlier_info = self.outlier_info
        
        # 전략이 없으면 기본값
        if strategy is None:
            strategy = {col: 'clip' for col in outlier_info.keys()}
        
        with logger.log_timer("이상치 처리"):
            for col, method in strategy.items():
                if col not in df_copy.columns:
                    continue
                
                if col not in outlier_info:
                    continue
                
                try:
                    df_copy[col] = self._handle_column_outliers(
                        df_copy[col], col, method
                    )
                except Exception as e:
                    logger.error(f"'{col}' 이상치 처리 실패: {e}")
        
        logger.info("✅ 이상치 처리 완료")
        return df_copy
    
    def _handle_column_outliers(self, series: pd.Series, col: str, 
                               method: str) -> pd.Series:
        """개별 컬럼 이상치 처리"""
        if col not in self.boundaries:
            logger.warning(f"'{col}' 경계값 없음")
            return series
        
        bounds = self.boundaries[col]
        
        if method == 'clip':
            # 경계값으로 클리핑
            if 'lower' in bounds and 'upper' in bounds:
                return series.clip(lower=bounds['lower'], upper=bounds['upper'])
        
        elif method == 'cap':
            # 백분위수로 캡핑
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            return series.clip(lower=lower, upper=upper)
        
        elif method == 'remove':
            # 이상치를 NaN으로 (나중에 결측치 처리)
            if 'lower' in bounds and 'upper' in bounds:
                mask = (series < bounds['lower']) | (series > bounds['upper'])
                series_copy = series.copy()
                series_copy[mask] = np.nan
                return series_copy
        
        elif method == 'fill_median':
            # 중앙값으로 대체
            if 'lower' in bounds and 'upper' in bounds:
                mask = (series < bounds['lower']) | (series > bounds['upper'])
                series_copy = series.copy()
                series_copy[mask] = series.median()
                return series_copy
        
        elif method == 'winsorize':
            # 윈저화 (1%, 99% 백분위수)
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            return series.clip(lower=lower, upper=upper)
        
        else:
            logger.warning(f"알 수 없는 방법: {method}, clip 사용")
            if 'lower' in bounds and 'upper' in bounds:
                return series.clip(lower=bounds['lower'], upper=bounds['upper'])
        
        return series
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """이상치 요약 테이블"""
        if not self.outlier_info:
            return pd.DataFrame()
        
        summary = []
        for col, info in self.outlier_info.items():
            summary.append({
                '컬럼': col,
                '이상치 개수': info['count'],
                '비율(%)': info['ratio'] * 100,
                '하한': info['bounds'].get('lower', 'N/A'),
                '상한': info['bounds'].get('upper', 'N/A')
            })
        
        return pd.DataFrame(summary)
    
    def save_boundaries(self, filepath: str):
        """경계값 저장"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'boundaries': self.boundaries,
                'outlier_info': self.outlier_info
            }, f)
        logger.info(f"💾 경계값 저장: {filepath}")
    
    def load_boundaries(self, filepath: str):
        """경계값 로드"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.boundaries = data['boundaries']
            self.outlier_info = data['outlier_info']
        logger.info(f"📂 경계값 로드: {filepath}")


if __name__ == "__main__":
    # 테스트
    detector = OutlierDetector()
    
    # 샘플 데이터 (의도적 이상치 포함)
    np.random.seed(42)
    df_test = pd.DataFrame({
        '정상': np.random.normal(100, 15, 100),
        '이상치포함': np.concatenate([
            np.random.normal(50, 10, 95),
            [200, 250, -50, 300, 350]  # 이상치
        ])
    })
    
    print("원본 데이터 통계:")
    print(df_test.describe())
    
    # 이상치 탐지
    outlier_info = detector.detect_outliers(df_test, method='iqr')
    
    print("\n이상치 탐지 결과:")
    for col, info in outlier_info.items():
        print(f"- {col}: {info['count']}개 ({info['ratio']:.1%})")
        print(f"  경계: [{info['bounds']['lower']:.1f}, {info['bounds']['upper']:.1f}]")
    
    # 이상치 처리
    df_handled = detector.handle_outliers(df_test, strategy={
        '이상치포함': 'clip'
    })
    
    print("\n처리 후 데이터 통계:")
    print(df_handled.describe())
    
    print("\n이상치 탐지기 테스트 완료")