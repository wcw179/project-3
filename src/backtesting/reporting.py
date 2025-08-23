"""
Reporting system for M5 Multi-Symbol Trend Bot backtests
Generates CSV, JSON, MD, and HTML reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

from src.data.database import TradingDatabase

logger = logging.getLogger(__name__)

class BacktestReporter:
    """Generate comprehensive backtest reports"""
    
    def __init__(self, output_dir: str = "artifacts/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = TradingDatabase()
        
    def generate_csv_report(self, backtest_results: Dict, filename: str = None) -> str:
        """Generate CSV report with trades and equity curve"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{backtest_results['symbol']}_{timestamp}.csv"
        
        csv_path = self.output_dir / filename
        
        # Prepare trades data
        trades_data = []
        for trade in backtest_results['trades']:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'symbol': trade.symbol,
                'direction': 'Long' if trade.direction == 1 else 'Short',
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct * 100,
                'holding_period': trade.holding_period,
                'rr_achieved': trade.rr_achieved,
                'exit_reason': trade.exit_reason,
                'lstm_prob': trade.lstm_prob,
                'xgb_score': trade.xgb_score
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        # Prepare equity curve data
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        
        # Write to CSV with multiple sheets (using Excel format)
        excel_path = csv_path.with_suffix('.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            trades_df.to_excel(writer, sheet_name='Trades', index=False)
            equity_df.to_excel(writer, sheet_name='Equity_Curve', index=False)
            
            # Metrics sheet
            metrics_df = pd.DataFrame([backtest_results['metrics']]).T
            metrics_df.columns = ['Value']
            metrics_df.to_excel(writer, sheet_name='Metrics')
        
        logger.info(f"CSV report saved to {excel_path}")
        return str(excel_path)
    
    def generate_json_report(self, backtest_results: Dict, filename: str = None) -> str:
        """Generate JSON report with all backtest data"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{backtest_results['symbol']}_{timestamp}.json"
        
        json_path = self.output_dir / filename
        
        # Prepare JSON-serializable data
        json_data = {
            'symbol': backtest_results['symbol'],
            'backtest_date': datetime.now().isoformat(),
            'config': {
                'start_date': backtest_results['config'].start_date.isoformat(),
                'end_date': backtest_results['config'].end_date.isoformat(),
                'initial_capital': backtest_results['config'].initial_capital,
                'risk_per_trade': backtest_results['config'].risk_per_trade,
                'min_rr_filter': backtest_results['config'].min_rr_filter,
                'lstm_threshold': backtest_results['config'].lstm_threshold,
                'xgb_threshold': backtest_results['config'].xgb_threshold
            },
            'metrics': backtest_results['metrics'],
            'trades': [
                {
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'symbol': trade.symbol,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'size': trade.size,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'holding_period': trade.holding_period,
                    'rr_achieved': trade.rr_achieved,
                    'exit_reason': trade.exit_reason,
                    'lstm_prob': trade.lstm_prob,
                    'xgb_score': trade.xgb_score
                }
                for trade in backtest_results['trades']
            ],
            'equity_curve': [
                {
                    'timestamp': point['timestamp'].isoformat(),
                    'equity': point['equity'],
                    'realized_pnl': point['realized_pnl'],
                    'unrealized_pnl': point['unrealized_pnl'],
                    'open_trades': point['open_trades'],
                    'closed_trades': point['closed_trades']
                }
                for point in backtest_results['equity_curve']
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"JSON report saved to {json_path}")
        return str(json_path)
    
    def generate_plots(self, backtest_results: Dict) -> Dict[str, str]:
        """Generate visualization plots"""
        plots = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        
        # 1. Equity Curve
        fig, ax = plt.subplots(figsize=fig_size)
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        ax.plot(equity_df['timestamp'], equity_df['equity'], linewidth=2, label='Equity')
        ax.axhline(y=backtest_results['config'].initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax.set_title(f"Equity Curve - {backtest_results['symbol']}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        equity_plot_path = self.output_dir / f"equity_curve_{backtest_results['symbol']}.png"
        plt.savefig(equity_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['equity_curve'] = str(equity_plot_path)
        
        # 2. Drawdown
        fig, ax = plt.subplots(figsize=fig_size)
        equity_values = equity_df['equity']
        rolling_max = equity_values.expanding().max()
        drawdown = (equity_values - rolling_max) / rolling_max * 100
        
        ax.fill_between(equity_df['timestamp'], drawdown, 0, alpha=0.7, color='red')
        ax.set_title(f"Drawdown - {backtest_results['symbol']}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        drawdown_plot_path = self.output_dir / f"drawdown_{backtest_results['symbol']}.png"
        plt.savefig(drawdown_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['drawdown'] = str(drawdown_plot_path)
        
        # 3. Trade Distribution
        if backtest_results['trades']:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            trades_df = pd.DataFrame([
                {
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct * 100,
                    'holding_period': trade.holding_period,
                    'rr_achieved': trade.rr_achieved,
                    'direction': 'Long' if trade.direction == 1 else 'Short'
                }
                for trade in backtest_results['trades']
            ])
            
            # PnL distribution
            ax1.hist(trades_df['pnl'], bins=30, alpha=0.7, edgecolor='black')
            ax1.set_title('PnL Distribution')
            ax1.set_xlabel('PnL ($)')
            ax1.set_ylabel('Frequency')
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # Return % distribution
            ax2.hist(trades_df['return_pct'], bins=30, alpha=0.7, edgecolor='black')
            ax2.set_title('Return % Distribution')
            ax2.set_xlabel('Return (%)')
            ax2.set_ylabel('Frequency')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # Holding period distribution
            ax3.hist(trades_df['holding_period'], bins=20, alpha=0.7, edgecolor='black')
            ax3.set_title('Holding Period Distribution')
            ax3.set_xlabel('Holding Period (bars)')
            ax3.set_ylabel('Frequency')
            
            # RR achieved distribution
            ax4.hist(trades_df['rr_achieved'], bins=20, alpha=0.7, edgecolor='black')
            ax4.set_title('Risk-Reward Achieved Distribution')
            ax4.set_xlabel('RR Achieved')
            ax4.set_ylabel('Frequency')
            ax4.axvline(x=1, color='green', linestyle='--', alpha=0.7, label='Break-even')
            ax4.legend()
            
            plt.tight_layout()
            
            trades_plot_path = self.output_dir / f"trade_analysis_{backtest_results['symbol']}.png"
            plt.savefig(trades_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['trade_analysis'] = str(trades_plot_path)
        
        return plots
    
    def generate_markdown_report(self, backtest_results: Dict, plots: Dict[str, str], filename: str = None) -> str:
        """Generate Markdown report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{backtest_results['symbol']}_{timestamp}.md"
        
        md_path = self.output_dir / filename
        
        metrics = backtest_results['metrics']
        config = backtest_results['config']
        
        md_content = f"""# Backtest Report: {backtest_results['symbol']}

## Summary
- **Symbol**: {backtest_results['symbol']}
- **Backtest Period**: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}
- **Initial Capital**: ${config.initial_capital:,.2f}
- **Final Equity**: ${metrics.get('final_equity', 0):,.2f}
- **Total Return**: {metrics.get('return_pct', 0):.2f}%

## Configuration
- **Risk per Trade**: {config.risk_per_trade * 100:.1f}%
- **Min RR Filter**: {config.min_rr_filter:.1f}
- **LSTM Threshold**: {config.lstm_threshold:.2f}
- **XGB Threshold**: {config.xgb_threshold:.2f}
- **EMA Confirmation**: {'Yes' if config.use_ema_confirmation else 'No'}
- **Black Swan Filter**: {'Yes' if config.use_black_swan_filter else 'No'}
- **HE Trailing Stop**: {'Yes' if config.use_he_trailing_stop else 'No'}

## Performance Metrics

### Profitability
- **Total Trades**: {metrics.get('total_trades', 0)}
- **Qualified Trades**: {metrics.get('qualified_trades', 0)}
- **Win Rate**: {metrics.get('win_rate', 0) * 100:.1f}%
- **Total PnL**: ${metrics.get('total_pnl', 0):,.2f}
- **Average Win**: ${metrics.get('avg_win', 0):,.2f}
- **Average Loss**: ${metrics.get('avg_loss', 0):,.2f}
- **Expectancy**: ${metrics.get('expectancy', 0):,.2f}
- **Profit Factor**: {metrics.get('profit_factor', 0):.2f}

### Risk Metrics
- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.2f}
- **Sortino Ratio**: {metrics.get('sortino_ratio', 0):.2f}
- **CAGR**: {metrics.get('cagr', 0) * 100:.2f}%
- **Maximum Drawdown**: {metrics.get('max_drawdown', 0) * 100:.2f}%

### Trade Characteristics
- **Average Holding Period**: {metrics.get('avg_holding_period', 0):.1f} bars
- **Average RR Achieved**: {metrics.get('avg_rr_achieved', 0):.2f}

## Exit Reasons
"""
        
        exit_reasons = metrics.get('exit_reasons', {})
        for reason, count in exit_reasons.items():
            md_content += f"- **{reason.replace('_', ' ').title()}**: {count} trades\n"
        
        md_content += "\n## Visualizations\n\n"
        
        for plot_name, plot_path in plots.items():
            plot_filename = Path(plot_path).name
            md_content += f"### {plot_name.replace('_', ' ').title()}\n"
            md_content += f"![{plot_name}]({plot_filename})\n\n"
        
        md_content += f"""
## Notes
- Only trades with RR ≥ {config.min_rr_filter} are included in performance metrics
- All times are in UTC
- Commission and slippage are included in PnL calculations
- Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report saved to {md_path}")
        return str(md_path)
    
    def generate_html_report(self, backtest_results: Dict, plots: Dict[str, str], filename: str = None) -> str:
        """Generate HTML report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{backtest_results['symbol']}_{timestamp}.html"
        
        html_path = self.output_dir / filename
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {{ symbol }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .summary { background: #e8f4fd; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .metric-value { font-size: 1.5em; font-weight: bold; color: #007bff; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .plot-container { text-align: center; margin: 20px 0; }
        .plot-container img { max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .config-table { background: #f8f9fa; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report: {{ symbol }}</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p><strong>Period:</strong> {{ start_date }} to {{ end_date }}</p>
            <p><strong>Total Return:</strong> <span class="{{ 'positive' if return_pct > 0 else 'negative' }}">{{ return_pct }}%</span></p>
            <p><strong>Sharpe Ratio:</strong> {{ sharpe_ratio }}</p>
            <p><strong>Max Drawdown:</strong> <span class="negative">{{ max_drawdown }}%</span></p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {{ 'positive' if total_pnl > 0 else 'negative' }}">${{ total_pnl }}</div>
                <div>Total PnL</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ win_rate }}%</div>
                <div>Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ total_trades }}</div>
                <div>Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ profit_factor }}</div>
                <div>Profit Factor</div>
            </div>
        </div>
        
        {% for plot_name, plot_path in plots.items() %}
        <div class="plot-container">
            <h3>{{ plot_name.replace('_', ' ').title() }}</h3>
            <img src="{{ plot_path }}" alt="{{ plot_name }}">
        </div>
        {% endfor %}
        
        <h2>Configuration</h2>
        <table class="config-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Initial Capital</td><td>${{ initial_capital }}</td></tr>
            <tr><td>Risk per Trade</td><td>{{ risk_per_trade }}%</td></tr>
            <tr><td>Min RR Filter</td><td>{{ min_rr_filter }}</td></tr>
            <tr><td>LSTM Threshold</td><td>{{ lstm_threshold }}</td></tr>
            <tr><td>XGB Threshold</td><td>{{ xgb_threshold }}</td></tr>
        </table>
        
        <p><em>Report generated on {{ generation_time }}</em></p>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        
        metrics = backtest_results['metrics']
        config = backtest_results['config']
        
        # Prepare plot paths (relative to HTML file)
        relative_plots = {name: Path(path).name for name, path in plots.items()}
        
        html_content = template.render(
            symbol=backtest_results['symbol'],
            start_date=config.start_date.strftime('%Y-%m-%d'),
            end_date=config.end_date.strftime('%Y-%m-%d'),
            return_pct=f"{metrics.get('return_pct', 0):.2f}",
            sharpe_ratio=f"{metrics.get('sharpe_ratio', 0):.2f}",
            max_drawdown=f"{metrics.get('max_drawdown', 0) * 100:.2f}",
            total_pnl=f"{metrics.get('total_pnl', 0):,.2f}",
            win_rate=f"{metrics.get('win_rate', 0) * 100:.1f}",
            total_trades=metrics.get('total_trades', 0),
            profit_factor=f"{metrics.get('profit_factor', 0):.2f}",
            initial_capital=f"{config.initial_capital:,.2f}",
            risk_per_trade=f"{config.risk_per_trade * 100:.1f}",
            min_rr_filter=config.min_rr_filter,
            lstm_threshold=config.lstm_threshold,
            xgb_threshold=config.xgb_threshold,
            plots=relative_plots,
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_path}")
        return str(html_path)
    
    def generate_complete_report(self, backtest_results: Dict) -> Dict[str, str]:
        """Generate all report formats"""
        logger.info(f"Generating complete report for {backtest_results['symbol']}")
        
        # Generate plots
        plots = self.generate_plots(backtest_results)
        
        # Generate all report formats
        reports = {
            'csv': self.generate_csv_report(backtest_results),
            'json': self.generate_json_report(backtest_results),
            'markdown': self.generate_markdown_report(backtest_results, plots),
            'html': self.generate_html_report(backtest_results, plots)
        }
        
        # Store report metadata in database
        self.store_report_metadata(backtest_results, reports)
        
        logger.info(f"Complete report generated for {backtest_results['symbol']}")
        return reports
    
    def store_report_metadata(self, backtest_results: Dict, report_paths: Dict[str, str]):
        """Store report metadata in database"""
        try:
            backtest_id = f"{backtest_results['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.db.insert_backtest_result(
                backtest_id=backtest_id,
                symbol=backtest_results['symbol'],
                strategy_config=json.dumps({
                    'start_date': backtest_results['config'].start_date.isoformat(),
                    'end_date': backtest_results['config'].end_date.isoformat(),
                    'initial_capital': backtest_results['config'].initial_capital,
                    'risk_per_trade': backtest_results['config'].risk_per_trade,
                    'min_rr_filter': backtest_results['config'].min_rr_filter
                }),
                metrics=json.dumps(backtest_results['metrics']),
                trades=json.dumps([{
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'rr_achieved': trade.rr_achieved
                } for trade in backtest_results['trades']]),
                equity_curve=json.dumps([{
                    'timestamp': point['timestamp'].isoformat(),
                    'equity': point['equity']
                } for point in backtest_results['equity_curve']]),
                report_path=report_paths.get('html', '')
            )
            
        except Exception as e:
            logger.error(f"Failed to store report metadata: {e}")
