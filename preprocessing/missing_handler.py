"""
EVA NEXIO DAIA - 결측치 처리
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.impute import SimpleImputer, KNNImputer
import yaml

from utils.logger import get_logger

logger = get_logger()


class MissingHandler:
    """결측치 처리 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.missing_config = self.config.get('preprocessing', {}).get('missing', {})
        self.imputers = {}  # 컬럼별 imputer 저장
        self.dropped_columns = []  # 삭제된 컬럼 기록
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def analyze_missing(self, df: pd.DataFrame) -> Dict:
        """결측치 분석"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_ratio = missing_count / len(df)
                missing_info[col] = {
                    'count': int(missing_count),
                    'ratio': float(missing_ratio),
                    'dtype': str(df[col].dtype),
                    'action': self._suggest_action(missing_ratio, df[col])
                }
        
        return missing_info
    
    def _suggest_action(self, missing_ratio: float, series: pd.Series) -> str:
        """결측치 처리 방법 제안"""
        threshold = self.missing_config.get('drop_threshold', 0.7)
        
        if missing_ratio >= threshold:
            return 'drop_column'
        
        if pd.api.types.is_numeric_dtype(series):
            if missing_ratio < 0.05:
                return 'fill_mean'
            else:
                return 'fill_median'
        else:
            return 'fill_mode'
    
    def handle_missing(self, df: pd.DataFrame, strategy: Dict = None, 
                      fit: bool = True) -> pd.DataFrame:
        """
        결측치 처리 (메인 함수)
        
        Args:
            df: 데이터프레임
            strategy: 컬럼별 처리 전략 {'col_name': 'method'}
            fit: True면 학습, False면 기존 imputer 사용
        """
        logger.log_step("결측치 처리", "START")
        df_copy = df.copy()
        
        # 전략이 없으면 자동 분석
        if strategy is None:
            missing_info = self.analyze_missing(df_copy)
            strategy = {col: info['action'] for col, info in missing_info.items()}
        
        with logger.log_timer("결측치 처리"):
            # 1. 삭제할 컬럼
            drop_cols = [col for col, method in strategy.items() 
                        if method == 'drop_column' and col in df_copy.columns]
            if drop_cols:
                logger.info(f"컬럼 삭제: {drop_cols}")
                df_copy = df_copy.drop(columns=drop_cols)
                self.dropped_columns.extend(drop_cols)
            
            # 2. 채우기
            for col, method in strategy.items():
                if col not in df_copy.columns or method == 'drop_column':
                    continue
                
                if df_copy[col].isnull().sum() == 0:
                    continue
                
                try:
                    df_copy[col] = self._fill_column(df_copy[col], method, col, fit)
                except Exception as e:
                    logger.warning(f"'{col}' 처리 실패 ({method}): {e}")
        
        remaining_missing = df_copy.isnull().sum().sum()
        logger.info(f"✅ 처리 완료 (남은 결측치: {remaining_missing}개)")
        
        return df_copy
    
    def _fill_column(self, series: pd.Series, method: str, col_name: str, 
                    fit: bool) -> pd.Series:
        """개별 컬럼 결측치 채우기"""
        
        if method == 'drop_row':
            return series.dropna()
        
        elif method == 'fill_mean':
            if fit:
                mean_val = series.mean()
                self.imputers[col_name] = {'method': 'mean', 'value': mean_val}
            else:
                mean_val = self.imputers.get(col_name, {}).get('value', series.mean())
            return series.fillna(mean_val)
        
        elif method == 'fill_median':
            if fit:
                median_val = series.median()
                self.imputers[col_name] = {'method': 'median', 'value': median_val}
            else:
                median_val = self.imputers.get(col_name, {}).get('value', series.median())
            return series.fillna(median_val)
        
        elif method == 'fill_mode':
            if fit:
                mode_val = series.mode()[0] if len(series.mode()) > 0 else series.value_counts().index[0]
                self.imputers[col_name] = {'method': 'mode', 'value': mode_val}
            else:
                mode_val = self.imputers.get(col_name, {}).get('value', 
                           series.mode()[0] if len(series.mode()) > 0 else None)
            return series.fillna(mode_val)
        
        elif method == 'fill_zero':
            return series.fillna(0)
        
        elif method == 'fill_unknown':
            return series.fillna('Unknown')
        
        elif method == 'forward_fill':
            return series.fillna(method='ffill')
        
        elif method == 'backward_fill':
            return series.fillna(method='bfill')
        
        else:
            logger.warning(f"알 수 없는 방법: {method}, median 사용")
            return series.fillna(series.median() if pd.api.types.is_numeric_dtype(series) 
                               else series.mode()[0])
    
    def get_imputer_info(self) -> Dict:
        """저장된 imputer 정보"""
        return {
            'imputers': self.imputers,
            'dropped_columns': self.dropped_columns
        }
    
    def save_imputers(self, filepath: str):
        """Imputer 저장"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'imputers': self.imputers,
                'dropped_columns': self.dropped_columns
            }, f)
        logger.info(f"💾 Imputer 저장: {filepath}")
    
    def load_imputers(self, filepath: str):
        """Imputer 로드"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.imputers = data['imputers']
            self.dropped_columns = data['dropped_columns']
        logger.info(f"📂 Imputer 로드: {filepath}")


if __name__ == "__main__":
    # 테스트
    handler = MissingHandler()
    
    # 샘플 데이터
    df_test = pd.DataFrame({
        '나이': [25, 30, None, 35, 40, None],
        '등급': ['VIP', None, '일반', 'VIP', None, '일반'],
        '구매액': [1000, 2000, None, None, 5000, 6000],
        '전부결측': [None] * 6
    })
    
    print("원본 데이터:")
    print(df_test)
    print(f"\n결측치: {df_test.isnull().sum().sum()}개")
    
    # 결측치 분석
    missing_info = handler.analyze_missing(df_test)
    print("\n결측치 분석:")
    for col, info in missing_info.items():
        print(f"- {col}: {info['ratio']:.1%} ({info['action']})")
    
    # 결측치 처리
    df_filled = handler.handle_missing(df_test)
    
    print("\n처리 후 데이터:")
    print(df_filled)
    print(f"\n남은 결측치: {df_filled.isnull().sum().sum()}개")
    
    print("\n결측치 핸들러 테스트 완료")