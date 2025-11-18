import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime
import threading
import time
from config import Config

class Dashboard:
    def __init__(self, engine):
        self.engine = engine
        self.config = Config()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.trade_history = []
        self.price_history = {symbol: [] for symbol in self.config.SYMBOLS}
        self.timestamps = []
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Настройка макета дашборда"""
        self.app.layout = dbc.Container([
            # Заголовок
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Bybit Arbitrage Bot Dashboard", 
                           className="text-center mb-4 text-primary"),
                    html.H5(f"{'TESTNET' if self.config.TESTNET else 'REAL'} MODE", 
                           className="text-center text-warning")
                ], width=12)
            ], className="mb-4"),
            
            # Статистика
            dbc.Row([
                dbc.Col(self._create_stat_card("💰 Total Profit", "profit_value", "0.00 USDT"), width=3),
                dbc.Col(self._create_stat_card("🎯 Total Trades", "trades_value", "0"), width=3),
                dbc.Col(self._create_stat_card("⚡ Avg Profit/Trade", "avg_profit_value", "0.00 USDT"), width=3),
                dbc.Col(self._create_stat_card("⏱️ Last Update", "time_value", "00:00:00"), width=3),
            ], className="mb-4"),
            
            # Графики
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='price-chart', config={'displayModeBar': False})
                ], width=8),
                dbc.Col([
                    dcc.Graph(id='profit-chart', config={'displayModeBar': False})
                ], width=4),
            ], className="mb-4"),
            
            # Спреды и сделки
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='spread-chart', config={'displayModeBar': False})
                ], width=6),
                dbc.Col([
                    html.H4("📈 Recent Trades", className="mb-3"),
                    dbc.Table(id='trades-table', bordered=True, hover=True, 
                             className="bg-dark text-light"),
                    html.Div(id='cooldown-status', className="mt-3")
                ], width=6),
            ], className="mb-4"),
            
            # Управление
            dbc.Row([
                dbc.Col([
                    html.H4("⚙️ Bot Controls", className="mb-3"),
                    dbc.ButtonGroup([
                        dbc.Button("▶️ Start", id="start-btn", color="success", className="me-2"),
                        dbc.Button("⏹️ Stop", id="stop-btn", color="danger", className="me-2"),
                        dbc.Button("🔄 Refresh", id="refresh-btn", color="info"),
                    ], className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Settings", className="card-title"),
                            dbc.Label("Update Interval (s):"),
                            dcc.Slider(
                                id='interval-slider',
                                min=1,
                                max=10,
                                step=1,
                                value=self.config.UPDATE_INTERVAL,
                                marks={i: str(i) for i in range(1, 11)}
                            ),
                            dbc.Label("Min Profit Threshold (%):"),
                            dcc.Slider(
                                id='profit-slider',
                                min=0.01,
                                max=1.0,
                                step=0.01,
                                value=self.config.MIN_PROFIT_PERCENT,
                                marks={0.1: '0.1%', 0.5: '0.5%', 1.0: '1.0%'}
                            ),
                            dbc.Label("Trade Amount (USDT):"),
                            dcc.Slider(
                                id='trade-amount-slider',
                                min=1,
                                max=100,
                                step=1,
                                value=self.config.TRADE_AMOUNT,
                                marks={10: '10', 50: '50', 100: '100'}
                            ),
                        ])
                    ], className="mt-3")
                ], width=12),
            ]),
            
            # Интервал обновления
            dcc.Interval(
                id='update-interval',
                interval=self.config.UPDATE_INTERVAL * 1000,
                n_intervals=0
            )
        ], fluid=True)
    
    def _create_stat_card(self, title, id, value):
        """Создание карточки статистики"""
        return dbc.Card([
            dbc.CardBody([
                html.H5(title, className="card-title text-muted"),
                html.H3(id=id, children=value, className="card-text text-success fw-bold")
            ])
        ], className="bg-dark border-primary")
    
    def setup_callbacks(self):
        """Настройка callback-функций для интерактивности"""
        
        @self.app.callback(
            [Output('profit_value', 'children'),
             Output('trades_value', 'children'),
             Output('avg_profit_value', 'children'),
             Output('time_value', 'children'),
             Output('price-chart', 'figure'),
             Output('profit-chart', 'figure'),
             Output('spread-chart', 'figure'),
             Output('trades-table', 'children'),
             Output('cooldown-status', 'children')],
            [Input('update-interval', 'n_intervals')]
        )
        def update_dashboard(n):
            # Обновление статистики
            profit = sum(trade.get('estimated_profit_usdt', 0) for trade in self.trade_history)
            trades_count = len(self.trade_history)
            avg_profit = profit / trades_count if trades_count > 0 else 0
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Обновление графиков
            price_fig = self._create_price_chart()
            profit_fig = self._create_profit_chart()
            spread_fig = self._create_spread_chart()
            
            # Обновление таблицы сделок
            trades_table = self._create_trades_table()
            cooldown_status = self._create_cooldown_status()
            
            return (
                f"{profit:.2f} USDT",
                str(trades_count),
                f"{avg_profit:.4f} USDT",
                current_time,
                price_fig,
                profit_fig,
                spread_fig,
                trades_table,
                cooldown_status
            )
        
        @self.app.callback(
            Output('update-interval', 'interval'),
            [Input('interval-slider', 'value')]
        )
        def update_interval(value):
            return value * 1000
        
        @self.app.callback(
            [Output('start-btn', 'disabled'),
             Output('stop-btn', 'disabled')],
            [Input('start-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks')],
            [State('start-btn', 'disabled'),
             State('stop-btn', 'disabled')]
        )
        def control_bot(start_clicks, stop_clicks, start_disabled, stop_disabled):
            # Логика управления ботом будет добавлена позже
            return start_disabled, stop_disabled
        
        @self.app.callback(
            Output('refresh-btn', 'n_clicks'),
            [Input('refresh-btn', 'n_clicks')]
        )
        def refresh_data(n_clicks):
            if n_clicks:
                # Принудительное обновление данных
                self.update_data()
            return 0
    
    def _create_price_chart(self):
        """Создание графика цен"""
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, symbol in enumerate(self.config.SYMBOLS):
            if symbol in self.price_history and len(self.price_history[symbol]) > 0:
                prices = [p['mid'] for p in self.price_history[symbol]]
                times = [p['timestamp'] for p in self.price_history[symbol]]
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=prices,
                    mode='lines+markers',
                    name=symbol,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    hovertemplate=f"{symbol}<br>Price: %{{y:.2f}} USDT<br>Time: %{{x}}<extra></extra>"
                ))
        
        fig.update_layout(
            title="💰 Real-time Prices",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_profit_chart(self):
        """Создание графика прибыли"""
        if not self.trade_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                title="📈 Cumulative Profit",
                template="plotly_dark",
                height=300
            )
            return fig
        
        cumulative_profit = []
        running_sum = 0
        timestamps = []
        
        for trade in self.trade_history:
            running_sum += trade.get('estimated_profit_usdt', 0)
            cumulative_profit.append(running_sum)
            timestamps.append(trade.get('timestamp', datetime.now()))
        
        fig = go.Figure()
        
        # Основная линия прибыли
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cumulative_profit,
            mode='lines+markers',
            name='Cumulative Profit',
            line=dict(color='#00FF00', width=3),
            marker=dict(size=6, color='#00FF00'),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        # Линия нулевой прибыли
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="📈 Cumulative Profit",
            xaxis_title="Time",
            yaxis_title="Profit (USDT)",
            template="plotly_dark",
            hovermode="x unified",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    def _create_spread_chart(self):
        """Создание графика спредов"""
        if not hasattr(self.engine, 'last_tickers'):
            fig = go.Figure()
            fig.add_annotation(
                text="Waiting for data...",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="gray")
            )
            fig.update_layout(
                title="📊 Spreads Analysis",
                template="plotly_dark",
                height=300
            )
            return fig
        
        symbols = []
        spreads = []
        colors = []
        
        for symbol, data in self.engine.last_tickers.items():
            if data['bid'] > 0 and data['ask'] > 0:
                spread = ((data['ask'] - data['bid']) / data['bid']) * 100
                symbols.append(symbol)
                spreads.append(spread)
                colors.append('#FF6B6B' if spread > 1 else '#4ECDC4')
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=symbols,
            y=spreads,
            marker_color=colors,
            text=[f"{spread:.2f}%" for spread in spreads],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Spread: %{y:.2f}%<extra></extra>"
        ))
        
        # Порог прибыльности
        fig.add_hline(
            y=self.config.MIN_PROFIT_PERCENT * 2,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Threshold: {self.config.MIN_PROFIT_PERCENT * 2:.2f}%",
            annotation_position="right"
        )
        
        fig.update_layout(
            title="📊 Spreads Analysis",
            xaxis_title="Symbols",
            yaxis_title="Spread (%)",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            yaxis=dict(range=[0, max(2, max(spreads) * 1.2) if spreads else 2])
        )
        
        return fig
    
    def _create_trades_table(self):
        """Создание таблицы последних сделок"""
        if not self.trade_history:
            return [
                html.Thead(html.Tr([html.Th("No trades executed yet")])),
                html.Tbody([])
            ]
        
        # Берем последние 10 сделок
        recent_trades = self.trade_history[-10:]
        
        table_header = [
            html.Thead(html.Tr([
                html.Th("Time"),
                html.Th("Symbol"),
                html.Th("Type"),
                html.Th("Profit (USDT)"),
                html.Th("Status")
            ]))
        ]
        
        table_body = []
        for trade in reversed(recent_trades):
            timestamp = trade.get('timestamp', datetime.now()).strftime("%H:%M:%S")
            symbol = trade.get('opportunity', {}).get('symbol', 'N/A')
            trade_type = trade.get('opportunity', {}).get('type', 'N/A').upper()
            profit = trade.get('estimated_profit_usdt', 0)
            status = "✅" if profit > 0 else "❌"
            
            row_color = "table-success" if profit > 0 else "table-danger"
            
            table_body.append(html.Tr([
                html.Td(timestamp),
                html.Td(symbol),
                html.Td(trade_type),
                html.Td(f"{profit:.4f}"),
                html.Td(status)
            ], className=row_color))
        
        return table_header + [html.Tbody(table_body)]
    
    def _create_cooldown_status(self):
        """Создание статуса кулдауна"""
        if not hasattr(self.engine, 'last_arbitrage_time') or not self.engine.last_arbitrage_time:
            return html.Div("No cooldowns active", className="text-muted")
        
        now = datetime.now()
        cooldown_items = []
        
        for symbol, last_time in self.engine.last_arbitrage_time.items():
            elapsed = (now - last_time).total_seconds()
            remaining = max(0, self.engine.cooldown_period - elapsed)
            
            if remaining > 0:
                progress = (elapsed / self.engine.cooldown_period) * 100
                cooldown_items.append(
                    dbc.Progress(
                        value=progress,
                        label=f"{symbol}: {remaining:.0f}s",
                        color="warning" if remaining < 60 else "info",
                        className="mb-2"
                    )
                )
        
        if not cooldown_items:
            return html.Div("✅ No active cooldowns", className="text-success")
        
        return html.Div([
            html.H5("⏳ Active Cooldowns", className="mb-2"),
            *cooldown_items
        ])
    
    def update_data(self):
        """Обновление данных для визуализации"""
        # Обновление истории цен
        if hasattr(self.engine, 'last_tickers'):
            current_time = datetime.now()
            
            for symbol, data in self.engine.last_tickers.items():
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                
                mid_price = (data['bid'] + data['ask']) / 2
                self.price_history[symbol].append({
                    'timestamp': current_time,
                    'mid': mid_price,
                    'bid': data['bid'],
                    'ask': data['ask']
                })
                
                # Ограничиваем историю последними 100 точками
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol].pop(0)
        
        # Обновление истории сделок
        if hasattr(self.engine, 'trade_history'):
            self.trade_history = self.engine.trade_history.copy()
    
    def run_dashboard(self, port=8050):
        """Запуск дашборда в отдельном потоке"""
        def run_app():
            self.app.run_server(
                host='0.0.0.0',
                port=port,
                debug=False,
                use_reloader=False
            )
        
        dashboard_thread = threading.Thread(target=run_app, daemon=True)
        dashboard_thread.start()
        print(f"📊 Dashboard started at http://localhost:{port}")
        
        # Запускаем цикл обновления данных
        self._start_data_update_loop()
    
    def _start_data_update_loop(self):
        """Цикл обновления данных для визуализации"""
        def update_loop():
            while True:
                try:
                    self.update_data()
                    time.sleep(1)  # Обновление данных каждую секунду
                except Exception as e:
                    print(f"Error updating dashboard data: {e}")
                    time.sleep(5)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()