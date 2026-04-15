"""
EVA NEXIO DAIA - 범주형 변수 인코딩
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import yaml

from utils.logger import get_logger

logger = get_logger()


class CategoricalEncoder:
    """범주형 변수 인코딩 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.encoding_config = self.config.get('preprocessing', {}).get('encoding', {})
        self.encoders = {}  # 컬럼별 인코더 저장
        self.encoding_map = {}  # 인코딩 매핑 저장
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def detect_categorical(self, df: pd.DataFrame, threshold: int = None) -> List[str]:
        """범주형 컬럼 자동 탐지"""
        if threshold is None:
            threshold = self.encoding_config.get('categorical_threshold', 10)
        
        categorical_cols = []
        
        for col in df.columns:
            # object 타입
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            # 수치형이지만 카테고리 같은 경우
            elif pd.api.types.is_numeric_dtype(df[col]):
                unique_count = df[col].nunique()
                if unique_count <= threshold and unique_count > 1:
                    categorical_cols.append(col)
        
        logger.info(f"범주형 컬럼 탐지: {categorical_cols}")
        return categorical_cols
    
    def suggest_encoding(self, df: pd.DataFrame, col: str) -> str:
        """인코딩 방법 제안"""
        unique_count = df[col].nunique()
        threshold = self.encoding_config.get('categorical_threshold', 10)
        
        if unique_count == 2:
            return 'label'  # 이진 변수는 레이블
        elif unique_count <= threshold:
            return 'onehot'  # 소수 카테고리는 원핫
        else:
            return 'label'  # 다수 카테고리는 레이블
    
    def encode(self, df: pd.DataFrame, encoding_plan: Dict = None, 
              fit: bool = True) -> pd.DataFrame:
        """
        범주형 변수 인코딩 (메인 함수)
        
        Args:
            df: 데이터프레임
            encoding_plan: 컬럼별 인코딩 방법 {'col': 'onehot' or 'label'}
            fit: True면 학습, False면 기존 인코더 사용
        """
        logger.log_step("범주형 인코딩", "START")
        df_copy = df.copy()
        
        # 인코딩 계획 타입 정규화
        # - None       → 자동 생성
        # - str        → 해당 방법을 모든 범주형 컬럼에 일괄 적용
        # - list/tuple → 해당 컬럼들에 자동 방법 적용
        # - dict       → 그대로 사용 (정상 케이스)
        if encoding_plan is None:
            categorical_cols = self.detect_categorical(df_copy)
            encoding_plan = {col: self.suggest_encoding(df_copy, col)
                             for col in categorical_cols}

        elif isinstance(encoding_plan, str):
            # ex) encoding_plan = "label" or "onehot"
            method_for_all = encoding_plan
            categorical_cols = self.detect_categorical(df_copy)
            encoding_plan = {col: method_for_all for col in categorical_cols}
            logger.info(f"encoding_plan이 str('{method_for_all}')로 전달됨 "
                        f"→ 모든 범주형 컬럼에 일괄 적용")

        elif isinstance(encoding_plan, (list, tuple)):
            # ex) encoding_plan = ["col_a", "col_b"]
            encoding_plan = {col: self.suggest_encoding(df_copy, col)
                             for col in encoding_plan if col in df_copy.columns}
            logger.info(f"encoding_plan이 list로 전달됨 → 자동 방법 적용")

        elif not isinstance(encoding_plan, dict):
            # 예상 외 타입 → 자동 생성으로 폴백
            logger.warning(f"encoding_plan 타입 미지원({type(encoding_plan).__name__}) "
                           f"→ 자동 생성으로 대체")
            categorical_cols = self.detect_categorical(df_copy)
            encoding_plan = {col: self.suggest_encoding(df_copy, col)
                             for col in categorical_cols}
        
        with logger.log_timer("범주형 인코딩"):
            for col, method in encoding_plan.items():
                if col not in df_copy.columns:
                    continue
                
                try:
                    if method == 'onehot':
                        df_copy = self._onehot_encode(df_copy, col, fit)
                    elif method == 'label':
                        df_copy[col] = self._label_encode(df_copy[col], col, fit)
                    else:
                        logger.warning(f"알 수 없는 인코딩: {method} for {col}")
                except Exception as e:
                    logger.error(f"'{col}' 인코딩 실패: {e}")
        
        logger.info(f"✅ 인코딩 완료 (Shape: {df_copy.shape})")
        return df_copy
    
    def _label_encode(self, series: pd.Series, col_name: str, fit: bool) -> pd.Series:
        """레이블 인코딩"""
        if fit:
            # 새로운 인코더 생성
            le = LabelEncoder()
            # 결측치 처리
            non_null_mask = series.notna()
            encoded = series.copy()
            
            if non_null_mask.sum() > 0:
                encoded.loc[non_null_mask] = le.fit_transform(series[non_null_mask].astype(str))
                self.encoders[col_name] = le
                self.encoding_map[col_name] = dict(zip(le.classes_, le.transform(le.classes_)))
                logger.debug(f"'{col_name}' 레이블 인코딩: {len(le.classes_)}개 클래스")
            
            return encoded
        else:
            # 기존 인코더 사용
            if col_name not in self.encoders:
                logger.warning(f"'{col_name}' 인코더 없음, 새로 생성")
                return self._label_encode(series, col_name, fit=True)
            
            le = self.encoders[col_name]
            non_null_mask = series.notna()
            encoded = series.copy()
            
            if non_null_mask.sum() > 0:
                # 새로운 카테고리 처리
                known_mask = series[non_null_mask].isin(le.classes_)
                if not known_mask.all():
                    unknown_count = (~known_mask).sum()
                    logger.warning(f"'{col_name}' 미지의 카테고리 {unknown_count}개 발견")
                    # 미지 카테고리는 가장 빈번한 값으로
                    mode_val = le.transform([le.classes_[0]])[0]
                    encoded.loc[non_null_mask & ~series.isin(le.classes_)] = mode_val
                
                valid_mask = non_null_mask & series.isin(le.classes_)
                encoded.loc[valid_mask] = le.transform(series[valid_mask].astype(str))
            
            return encoded
    
    def _onehot_encode(self, df: pd.DataFrame, col: str, fit: bool) -> pd.DataFrame:
        """원핫 인코딩"""
        if fit:
            # 고유값 추출
            unique_vals = df[col].dropna().unique()
            self.encoders[col] = {
                'type': 'onehot',
                'categories': list(unique_vals)
            }
            logger.debug(f"'{col}' 원핫 인코딩: {len(unique_vals)}개 더미 변수")
        
        categories = self.encoders.get(col, {}).get('categories', df[col].dropna().unique())
        
        # 더미 변수 생성
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
        
        # 학습시 보지 못한 카테고리 처리
        expected_cols = [f"{col}_{cat}" for cat in categories]
        for exp_col in expected_cols:
            if exp_col not in dummies.columns:
                dummies[exp_col] = 0
        
        # 불필요한 컬럼 제거
        dummies = dummies[[c for c in expected_cols if c in dummies.columns]]
        
        # 원본 컬럼 삭제 후 더미 추가
        df_result = df.drop(columns=[col])
        df_result = pd.concat([df_result, dummies], axis=1)
        
        return df_result
    
    def get_encoding_info(self) -> Dict:
        """인코딩 정보 조회"""
        info = {}
        for col, encoder in self.encoders.items():
            if isinstance(encoder, LabelEncoder):
                info[col] = {
                    'type': 'label',
                    'classes': list(encoder.classes_),
                    'mapping': self.encoding_map.get(col, {})
                }
            elif isinstance(encoder, dict) and encoder.get('type') == 'onehot':
                info[col] = {
                    'type': 'onehot',
                    'categories': encoder['categories']
                }
        return info
    
    def reverse_encoding(self, df: pd.DataFrame, col: str) -> pd.Series:
        """인코딩 복원 (레이블 인코딩만 가능)"""
        if col not in self.encoders:
            logger.warning(f"'{col}' 인코더 없음")
            return df[col]
        
        encoder = self.encoders[col]
        if isinstance(encoder, LabelEncoder):
            return pd.Series(encoder.inverse_transform(df[col].astype(int)), 
                           index=df.index)
        else:
            logger.warning("원핫 인코딩은 복원 불가")
            return df[col]
    
    def save_encoders(self, filepath: str):
        """인코더 저장"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'encoders': self.encoders,
                'encoding_map': self.encoding_map
            }, f)
        logger.info(f"💾 인코더 저장: {filepath}")
    
    def load_encoders(self, filepath: str):
        """인코더 로드"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.encoders = data['encoders']
            self.encoding_map = data['encoding_map']
        logger.info(f"📂 인코더 로드: {filepath}")


if __name__ == "__main__":
    # 테스트
    encoder = CategoricalEncoder()
    
    # 샘플 데이터
    df_test = pd.DataFrame({
        '고객ID': range(1, 11),
        '등급': ['VIP', '일반', 'VIP', '신규', '일반', 'VIP', '일반', 'VIP', '신규', '일반'],
        '지역': ['서울', '부산', '서울', '대구', '부산', '서울', '대구', '서울', '부산', '대구'],
        '성별': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']
    })
    
    print("원본 데이터:")
    print(df_test)
    
    # 인코딩
    df_encoded = encoder.encode(df_test)
    
    print("\n인코딩 후:")
    print(df_encoded)
    print(f"\nShape: {df_test.shape} → {df_encoded.shape}")
    
    # 인코딩 정보
    print("\n인코딩 정보:")
    info = encoder.get_encoding_info()
    for col, details in info.items():
        print(f"- {col}: {details['type']}")
    
    print("\n인코더 테스트 완료")