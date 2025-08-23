"""
Hierarchical Extremes (HE) implementation for M5 Multi-Symbol Trend Bot
Identifies multi-level support/resistance levels for trailing stops
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class HierarchicalExtremes:
    """Hierarchical Extremes detection system"""
    
    def __init__(self, levels: int = 3, atr_lookback: int = 1440):
        self.levels = levels
        self.atr_lookback = atr_lookback
        
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR for distance thresholds"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def find_local_extremes(self, prices: pd.Series, order: int = 5) -> Tuple[pd.Series, pd.Series]:
        """Find local highs and lows using scipy"""
        # Find local maxima (highs)
        high_indices = argrelextrema(prices.values, np.greater, order=order)[0]
        highs = pd.Series(index=prices.index[high_indices], data=prices.iloc[high_indices])
        
        # Find local minima (lows)
        low_indices = argrelextrema(prices.values, np.less, order=order)[0]
        lows = pd.Series(index=prices.index[low_indices], data=prices.iloc[low_indices])
        
        return highs, lows
    
    def filter_extremes_by_atr(self, extremes: pd.Series, prices: pd.Series, atr: pd.Series, 
                              min_atr_distance: float = 1.0) -> pd.Series:
        """Filter extremes that are too close together based on ATR"""
        if len(extremes) <= 1:
            return extremes
        
        filtered_extremes = []
        last_extreme = None
        
        for timestamp, price in extremes.items():
            if last_extreme is None:
                filtered_extremes.append((timestamp, price))
                last_extreme = (timestamp, price)
                continue
            
            # Get ATR at current time, handling potential duplicates by taking the first value
            current_atr_val = atr.loc[timestamp] if timestamp in atr.index else atr.iloc[-1]
            if isinstance(current_atr_val, pd.Series):
                current_atr_val = current_atr_val.iloc[0] # Handle duplicate timestamps

            # Check distance from last extreme
            price_distance = abs(price - last_extreme[1])
            # Ensure current_atr is a scalar before comparison
            atr_val = current_atr_val

            if atr_val > 0:
                atr_distance = price_distance / atr_val
                if atr_distance >= min_atr_distance:
                    filtered_extremes.append((timestamp, price))
                    last_extreme = (timestamp, price)
            elif price_distance > 0: # Handle zero ATR case
                filtered_extremes.append((timestamp, price))
                last_extreme = (timestamp, price)
        
        if filtered_extremes:
            timestamps, prices = zip(*filtered_extremes)
            return pd.Series(index=timestamps, data=prices)
        else:
            return pd.Series(dtype=float)
    
    def cluster_extremes(self, extremes: pd.Series, n_clusters: int) -> Dict[int, List[Tuple]]:
        """Cluster extremes by price level"""
        if len(extremes) < n_clusters:
            # Not enough extremes for clustering
            clusters = {}
            for i, (timestamp, price) in enumerate(extremes.items()):
                clusters[i] = [(timestamp, price)]
            return clusters
        
        # Prepare data for clustering
        prices = extremes.values.reshape(-1, 1)
        timestamps = extremes.index
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(prices)
        
        # Group extremes by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((timestamps[i], extremes.iloc[i]))
        
        return clusters
    
    def calculate_level_strength(self, cluster: List[Tuple], prices: pd.Series) -> float:
        """Calculate strength of a support/resistance level"""
        if not cluster:
            return 0.0
        
        # Number of touches
        touch_count = len(cluster)
        
        # Time span of the level
        timestamps = [item[0] for item in cluster]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600  # Hours
        
        # Price consistency (lower std = higher strength)
        level_prices = [item[1] for item in cluster]
        price_std = np.std(level_prices) if len(level_prices) > 1 else 0
        price_consistency = 1 / (1 + price_std)
        
        # Recency factor (more recent = higher strength)
        latest_timestamp = max(timestamps)
        current_timestamp = prices.index[-1]
        hours_since_latest = (current_timestamp - latest_timestamp).total_seconds() / 3600
        recency_factor = np.exp(-hours_since_latest / 168)  # Decay over 1 week
        
        # Combined strength score
        strength = (touch_count * 0.4 + 
                   min(time_span / 24, 10) * 0.2 +  # Cap at 10 days
                   price_consistency * 0.2 + 
                   recency_factor * 0.2)
        
        return strength
    
    def identify_hierarchical_levels(self, ohlc: pd.DataFrame) -> Dict[int, Dict]:
        """Identify hierarchical support/resistance levels"""
        high_prices = ohlc['high']
        low_prices = ohlc['low']
        close_prices = ohlc['close']
        
        # Calculate ATR for filtering
        atr = self.calculate_atr(ohlc['high'], ohlc['low'], ohlc['close'])
        
        hierarchical_levels = {}
        
        for level in range(1, self.levels + 1):
            # Adjust parameters for each level
            order = level * 3  # More strict for higher levels
            min_atr_distance = level * 0.5  # Larger distance for higher levels
            
            # Find resistance levels (highs)
            raw_highs, _ = self.find_local_extremes(high_prices, order=order)
            filtered_highs = self.filter_extremes_by_atr(raw_highs, high_prices, atr, min_atr_distance)
            
            # Find support levels (lows)
            _, raw_lows = self.find_local_extremes(low_prices, order=order)
            filtered_lows = self.filter_extremes_by_atr(raw_lows, low_prices, atr, min_atr_distance)
            
            # Cluster extremes for this level
            n_clusters = max(2, min(8, len(filtered_highs) // 2)) if len(filtered_highs) > 0 else 0
            resistance_clusters = self.cluster_extremes(filtered_highs, n_clusters) if n_clusters > 0 else {}
            
            n_clusters = max(2, min(8, len(filtered_lows) // 2)) if len(filtered_lows) > 0 else 0
            support_clusters = self.cluster_extremes(filtered_lows, n_clusters) if n_clusters > 0 else {}
            
            # Calculate level information
            resistance_levels = []
            for cluster_id, cluster in resistance_clusters.items():
                level_price = np.mean([item[1] for item in cluster])
                strength = self.calculate_level_strength(cluster, close_prices)
                resistance_levels.append({
                    'price': level_price,
                    'strength': strength,
                    'touches': len(cluster),
                    'type': 'resistance',
                    'cluster_id': cluster_id,
                    'extremes': cluster
                })
            
            support_levels = []
            for cluster_id, cluster in support_clusters.items():
                level_price = np.mean([item[1] for item in cluster])
                strength = self.calculate_level_strength(cluster, close_prices)
                support_levels.append({
                    'price': level_price,
                    'strength': strength,
                    'touches': len(cluster),
                    'type': 'support',
                    'cluster_id': cluster_id,
                    'extremes': cluster
                })
            
            # Sort by strength
            resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            hierarchical_levels[level] = {
                'resistance': resistance_levels,
                'support': support_levels,
                'total_levels': len(resistance_levels) + len(support_levels)
            }
        
        return hierarchical_levels
    
    def get_nearest_levels(self, hierarchical_levels: Dict[int, Dict], current_price: float, 
                          level: int = 1, max_distance_atr: float = 5.0, 
                          current_atr: float = None) -> Dict:
        """Get nearest support/resistance levels for current price"""
        if level not in hierarchical_levels:
            return {'nearest_support': None, 'nearest_resistance': None}
        
        level_data = hierarchical_levels[level]
        
        # Find nearest support (below current price)
        nearest_support = None
        min_support_distance = float('inf')
        
        for support in level_data['support']:
            if support['price'] < current_price:
                distance = current_price - support['price']
                if current_atr and distance / current_atr <= max_distance_atr:
                    if distance < min_support_distance:
                        min_support_distance = distance
                        nearest_support = support
                elif not current_atr and distance < min_support_distance:
                    min_support_distance = distance
                    nearest_support = support
        
        # Find nearest resistance (above current price)
        nearest_resistance = None
        min_resistance_distance = float('inf')
        
        for resistance in level_data['resistance']:
            if resistance['price'] > current_price:
                distance = resistance['price'] - current_price
                if current_atr and distance / current_atr <= max_distance_atr:
                    if distance < min_resistance_distance:
                        min_resistance_distance = distance
                        nearest_resistance = resistance
                elif not current_atr and distance < min_resistance_distance:
                    min_resistance_distance = distance
                    nearest_resistance = resistance
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance': min_support_distance if nearest_support else None,
            'resistance_distance': min_resistance_distance if nearest_resistance else None
        }
    
    def calculate_trailing_stop_level(self, hierarchical_levels: Dict[int, Dict], 
                                     current_price: float, position_type: str,
                                     current_atr: float = None, level: int = 1) -> Optional[float]:
        """Calculate trailing stop level using Lv1 HE"""
        nearest_levels = self.get_nearest_levels(hierarchical_levels, current_price, level, 
                                               current_atr=current_atr)
        
        if position_type.lower() == 'long':
            # For long positions, use nearest support as trailing stop
            if nearest_levels['nearest_support']:
                support_level = nearest_levels['nearest_support']['price']
                # Add small buffer below support
                buffer = current_atr * 0.2 if current_atr else support_level * 0.001
                return support_level - buffer
        
        elif position_type.lower() == 'short':
            # For short positions, use nearest resistance as trailing stop
            if nearest_levels['nearest_resistance']:
                resistance_level = nearest_levels['nearest_resistance']['price']
                # Add small buffer above resistance
                buffer = current_atr * 0.2 if current_atr else resistance_level * 0.001
                return resistance_level + buffer
        
        return None
    
    def get_level_features(self, hierarchical_levels: Dict[int, Dict], current_price: float,
                          current_atr: float = None) -> Dict:
        """Extract features from hierarchical levels for ML models"""
        features = {}
        
        for level in range(1, self.levels + 1):
            if level not in hierarchical_levels:
                continue
            
            nearest = self.get_nearest_levels(hierarchical_levels, current_price, level, 
                                            current_atr=current_atr)
            
            # Support features
            if nearest['nearest_support']:
                support = nearest['nearest_support']
                features[f'support_distance_l{level}'] = nearest['support_distance']
                features[f'support_strength_l{level}'] = support['strength']
                features[f'support_touches_l{level}'] = support['touches']
                if current_atr:
                    features[f'support_distance_atr_l{level}'] = nearest['support_distance'] / current_atr
            else:
                features[f'support_distance_l{level}'] = np.nan
                features[f'support_strength_l{level}'] = 0
                features[f'support_touches_l{level}'] = 0
                features[f'support_distance_atr_l{level}'] = np.nan
            
            # Resistance features
            if nearest['nearest_resistance']:
                resistance = nearest['nearest_resistance']
                features[f'resistance_distance_l{level}'] = nearest['resistance_distance']
                features[f'resistance_strength_l{level}'] = resistance['strength']
                features[f'resistance_touches_l{level}'] = resistance['touches']
                if current_atr:
                    features[f'resistance_distance_atr_l{level}'] = nearest['resistance_distance'] / current_atr
            else:
                features[f'resistance_distance_l{level}'] = np.nan
                features[f'resistance_strength_l{level}'] = 0
                features[f'resistance_touches_l{level}'] = 0
                features[f'resistance_distance_atr_l{level}'] = np.nan
            
            # Level density (how many levels are nearby)
            level_data = hierarchical_levels[level]
            nearby_levels = 0
            max_distance = current_atr * 3 if current_atr else current_price * 0.01
            
            for support in level_data['support']:
                if abs(support['price'] - current_price) <= max_distance:
                    nearby_levels += 1
            
            for resistance in level_data['resistance']:
                if abs(resistance['price'] - current_price) <= max_distance:
                    nearby_levels += 1
            
            features[f'level_density_l{level}'] = nearby_levels
        
        return features
    
    def update_hierarchical_levels(self, existing_levels: Dict[int, Dict], 
                                  new_ohlc: pd.DataFrame) -> Dict[int, Dict]:
        """Update existing hierarchical levels with new data"""
        # For simplicity, recalculate all levels with new data
        # In production, you might want to implement incremental updates
        return self.identify_hierarchical_levels(new_ohlc)
    
    def validate_levels(self, hierarchical_levels: Dict[int, Dict], ohlc: pd.DataFrame) -> Dict:
        """Validate the quality of identified levels"""
        validation_results = {
            'total_levels': 0,
            'avg_strength': 0,
            'level_distribution': {},
            'quality_score': 0
        }
        
        all_strengths = []
        
        for level, level_data in hierarchical_levels.items():
            level_count = level_data['total_levels']
            validation_results['total_levels'] += level_count
            validation_results['level_distribution'][f'level_{level}'] = level_count
            
            # Collect strength scores
            for support in level_data['support']:
                all_strengths.append(support['strength'])
            for resistance in level_data['resistance']:
                all_strengths.append(resistance['strength'])
        
        if all_strengths:
            validation_results['avg_strength'] = np.mean(all_strengths)
            validation_results['quality_score'] = min(100, validation_results['avg_strength'] * 20)
        
        return validation_results
