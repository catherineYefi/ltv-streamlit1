import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка matplotlib для темной темы
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0E0E0E'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['axes.edgecolor'] = '#2E2E2E'
plt.rcParams['grid.color'] = '#333333'
plt.rcParams['text.color'] = '#e6e6e6'
plt.rcParams['axes.labelcolor'] = '#e6e6e6'
plt.rcParams['xtick.color'] = '#e6e6e6'
plt.rcParams['ytick.color'] = '#e6e6e6'

# =========================
# Брендинг Ultima
# =========================
ULTIMA_GOLD = "#F9B233"
ULTIMA_DARK = "#0E0E0E"
ULTIMA_GRAY = "#2E2E2E"
ULTIMA_SUCCESS = "#3bd16f"
ULTIMA_WARNING = "#ffcf3a"
ULTIMA_ERROR = "#ff5f73"
ULTIMA_INFO = "#5ab0ff"

ENHANCED_HEADER_HTML = f"""
<div style="display:flex;align-items:center;gap:20px;margin-bottom:12px;padding:16px;background:linear-gradient(135deg,{ULTIMA_DARK},{ULTIMA_GRAY});border-radius:16px">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/68a5d644d41e00d772823934/bmu2UTnqh39vYO0wRV718.png" style="width:64px;height:64px"/> 
  <div style="flex:1">
    <div style="font-size:32px;font-weight:800;color:{ULTIMA_GOLD};line-height:1.2;margin-bottom:4px">
      🚀 ULTIMA — Advanced LTV Analytics
    </div>
    <div style="color:#aaa;font-size:14px">
      Продвинутая unit-экономика • Сценарное моделирование • Детальная аналитика
    </div>
  </div>
  <div style="text-align:right;color:#666;font-size:12px">
    v2.0 Enhanced<br>
    {datetime.now().strftime('%d.%m.%Y %H:%M')}
  </div>
</div>
"""

ENHANCED_KPI_CSS = f"""
<style>
.kpi-grid {{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
  gap:16px; margin:12px 0;
}}
.kpi-card {{
  background:linear-gradient(135deg,{ULTIMA_DARK},{ULTIMA_GRAY});
  border:2px solid {ULTIMA_GRAY};
  border-radius:16px; padding:20px; text-align:center;
  transition: all 0.3s ease; position: relative; overflow: hidden;
  color:#fff;
}}
.kpi-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: var(--accent-color, {ULTIMA_GOLD});
}}
.kpi-label {{ color:#fff !important; font-size:13px; font-weight:500; margin-bottom:8px; }}
.kpi-value {{ color:#fff !important; font-size:24px; font-weight:800; margin:8px 0; }}
.kpi-change {{ color:#fff !important; font-size:11px; opacity:0.8; margin-top:4px; }}
.kpi-good    {{ --accent-color: {ULTIMA_SUCCESS}; }}
.kpi-warn    {{ --accent-color: {ULTIMA_WARNING}; }}
.kpi-bad     {{ --accent-color: {ULTIMA_ERROR}; }}
.kpi-neutral {{ --accent-color: {ULTIMA_INFO}; }}

.insights-panel {{
  background:{ULTIMA_DARK}; border:1px solid {ULTIMA_GRAY}; border-radius:12px;
  padding:16px; margin:12px 0;
}}
.insights-panel h4 {{
  color:#fff !important;
  font-weight:700;
}}
.insight-item {{ 
  display:flex; align-items:center; gap:12px; padding:8px 0;
  border-bottom:1px solid #333;
}}
.insight-item:last-child {{ border-bottom:none; }}
.insight-icon {{ font-size:20px; }}
.insight-text {{ flex:1; font-size:14px; line-height:1.4; color: #fff !important; }}

.warning-panel {{
  color: #fff !important;
  background: #2a2a1a;
  padding: 12px;
  border-radius: 8px;
  margin: 8px 0;
}}
.warning-panel h5 {{
  color: #fff !important;
  font-weight: 700;
}}
.warning-panel ul {{
  color: #fff !important;
}}
.warning-panel li {{
  color: #fff !important;
}}
</style>
"""

# =========================
# Улучшенная модель расчетов
# =========================
class EnhancedLTVModel:
    def __init__(self):
        self.industry_benchmarks = {
            "SaaS": {"ltv_cac_min": 3.0, "payback_max": 12, "churn_typical": 5},
            "E-commerce": {"ltv_cac_min": 2.5, "payback_max": 6, "churn_typical": 15},
            "Marketplace": {"ltv_cac_min": 2.0, "payback_max": 8, "churn_typical": 12},
            "Fintech": {"ltv_cac_min": 4.0, "payback_max": 18, "churn_typical": 8},
            "Услуги (салоны, фитнес, обучение)": {"ltv_cac_min": 2.0, "payback_max": 9, "churn_typical": 20,},
        }
    
    def validate_inputs(self, params):
        warnings = []
        errors = []
        if params['avg_check'] <= 0:
            errors.append("Средний чек должен быть положительным")
        if params['cac'] <= 0:
            errors.append("CAC должен быть положительным")
        if params['monthly_churn_pct'] >= 50:
            warnings.append("⚠️ Очень высокий отток - проверьте корректность данных")
        if params['margin_pct'] < 20:
            warnings.append("⚠️ Низкая маржинальность может негативно влиять на LTV")
        if params['avg_check'] < params['cac']:
            warnings.append("⚠️ Средний чек меньше CAC - окупаемость под вопросом")
        return warnings, errors

    def calculate_scenarios(self, base_params):
        scenarios = {}
        multipliers = {
            "Пессимистичный": {"churn": 1.5, "margin": 0.8, "check": 0.9},
            "Базовый": {"churn": 1.0, "margin": 1.0, "check": 1.0},
            "Оптимистичный": {"churn": 0.7, "margin": 1.2, "check": 1.1}
        }
        for scenario_name, mult in multipliers.items():
            params = base_params.copy()
            params['monthly_churn_pct'] *= mult['churn']
            params['margin_pct'] *= mult['margin']
            params['avg_check'] *= mult['check']
            ltv_data = self.compute_enhanced_ltv(params)
            scenarios[scenario_name] = ltv_data
        return scenarios

    def compute_enhanced_ltv(self, params):
        monthly_revenue = (params['avg_check'] * params['purchases_per_year']) / 12.0
        monthly_margin = monthly_revenue * (params['margin_pct'] / 100.0)
        survival_rate = 1.0 - params['monthly_churn_pct'] / 100.0
        monthly_discount = 1.0 / ((1.0 + params['discount_rate_pct'] / 100.0) ** (1.0 / 12.0))
        months = np.arange(1, params['horizon_months'] + 1)
        survival_curve = survival_rate ** (months - 1)
        discount_curve = monthly_discount ** months
        seasonality = 1 + 0.05 * np.sin(2 * np.pi * months / 12)
        monthly_cf = monthly_margin * seasonality * survival_curve * discount_curve
        cumulative_cf = np.cumsum(monthly_cf)
        ltv = float(np.sum(monthly_cf))
        ltv_cac = (ltv / params['cac']) if params['cac'] > 0 else 0
        roi = ((ltv - params['cac']) / params['cac']) if params['cac'] > 0 else 0
        payback_month = None
        if np.any(cumulative_cf >= params['cac']):
            payback_month = int(months[cumulative_cf >= params['cac']][0])
        customer_lifetime = 1 / (params['monthly_churn_pct'] / 100) if params['monthly_churn_pct'] > 0 else np.inf
        monthly_retention = (1 - params['monthly_churn_pct'] / 100) * 100
        annual_retention = ((1 - params['monthly_churn_pct'] / 100) ** 12) * 100
        confidence_score = min(100, max(0, (ltv_cac - 1) * 25)) if ltv_cac > 0 else 0
        return {
            'ltv': ltv, 'ltv_cac': ltv_cac, 'roi': roi,
            'payback_month': payback_month, 'customer_lifetime': customer_lifetime,
            'monthly_retention': monthly_retention, 'annual_retention': annual_retention,
            'confidence_score': confidence_score,
            'monthly_cf': monthly_cf, 'cumulative_cf': cumulative_cf,
            'months': months, 'survival_curve': survival_curve
        }

    def generate_insights(self, ltv_data, params, industry="SaaS"):
        insights = []
        benchmark = self.industry_benchmarks.get(industry, self.industry_benchmarks["SaaS"])
        if ltv_data['ltv_cac'] < benchmark['ltv_cac_min']:
            insights.append({
                "icon": "⚠️", "type": "warning",
                "text": f"LTV/CAC ниже отраслевого минимума ({benchmark['ltv_cac_min']}). Необходимо увеличить LTV или снизить CAC."
            })
        else:
            insights.append({
                "icon": "✅", "type": "success", 
                "text": f"LTV/CAC соответствует отраслевым стандартам ({ltv_data['ltv_cac']:.1f})."
            })
        if ltv_data['payback_month'] and ltv_data['payback_month'] <= benchmark['payback_max']:
            insights.append({
                "icon": "🚀", "type": "success",
                "text": f"Отличный payback период ({ltv_data['payback_month']} мес.) - быстрая окупаемость."
            })
        elif ltv_data['payback_month']:
            insights.append({
                "icon": "⏳", "type": "warning",
                "text": f"Payback период ({ltv_data['payback_month']} мес.) выше рекомендуемого ({benchmark['payback_max']} мес.)."
            })
        if params['monthly_churn_pct'] > benchmark['churn_typical'] * 1.5:
            insights.append({
                "icon": "📉", "type": "error",
                "text": f"Критически высокий отток ({params['monthly_churn_pct']:.1f}% в месяц). Срочно требуется работа с retention."
            })
        if ltv_data['ltv_cac'] < 3:
            if params['margin_pct'] < 30:
                insights.append({
                    "icon": "💡", "type": "info",
                    "text": "Рекомендация: увеличить маржинальность через апсейл или снижение затрат."
                })
            if params['monthly_churn_pct'] > 10:
                insights.append({
                    "icon": "💡", "type": "info", 
                    "text": "Рекомендация: инвестировать в программы лояльности для снижения оттока."
                })
        return insights

model = EnhancedLTVModel()

def create_scenarios_chart(scenarios):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Пессимистичный": "#ff5f73", "Базовый": "#5ab0ff", "Оптимистичный": "#3bd16f"}
    for scenario_name, data in scenarios.items():
        ax.plot(data['months'], data['cumulative_cf'], 
               label=f"CF {scenario_name}", color=colors[scenario_name], linewidth=3)
    ax.axhline(y=15000, color="#ffcf3a", linestyle="--", linewidth=2, label="CAC")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Кумулятивный CF (₽)")
    ax.set_title("📊 Кумулятивный Cash Flow по сценариям", fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_survival_chart(scenarios):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"Пессимистичный": "#ff5f73", "Базовый": "#5ab0ff", "Оптимистичный": "#3bd16f"}
    for scenario_name, data in scenarios.items():
        ax.plot(data['months'], data['survival_curve'] * 100, 
               label=f"Выживаемость {scenario_name}", color=colors[scenario_name], linewidth=3)
        ax.fill_between(data['months'], data['survival_curve'] * 100, alpha=0.2, color=colors[scenario_name])
    ax.set_xlabel("Месяц")
    ax.set_ylabel("% клиентов")
    ax.set_title("📈 Кривые выживаемости клиентов", fontsize=14, pad=20)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_sensitivity_chart(base_params):
    fig, ax = plt.subplots(figsize=(12, 6))
    params_to_test = ['avg_check', 'margin_pct', 'monthly_churn_pct', 'purchases_per_year']
    param_names = ['Средний чек', 'Маржа %', 'Отток %', 'Покупок/год']
    variations = np.linspace(0.5, 1.5, 11)
    colors = ['#5ab0ff', '#3bd16f', '#ff5f73', '#ffcf3a']
    for i, param in enumerate(params_to_test):
        ltv_cac_values = []
        for mult in variations:
            test_params = base_params.copy()
            test_params[param] *= mult
            ltv_data = model.compute_enhanced_ltv(test_params)
            ltv_cac_values.append(ltv_data['ltv_cac'])
        ax.plot(variations * 100, ltv_cac_values, 
               label=param_names[i], color=colors[i], linewidth=3, marker='o', markersize=4)
    ax.axhline(y=3, color="#ffcf3a", linestyle="--", linewidth=2, alpha=0.8, label="Целевой LTV/CAC = 3.0")
    ax.axhline(y=1, color="#ff5f73", linestyle=":", linewidth=2, alpha=0.8, label="Критический уровень = 1.0")
    ax.set_xlabel("% изменения параметра")
    ax.set_ylabel("LTV/CAC")
    ax.set_title("🎯 Анализ чувствительности LTV/CAC", fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_enhanced_kpi_cards(scenarios):
    data = scenarios["Базовый"]
    def get_kpi_class(metric, value):
        if metric == "ltv_cac":
            return "kpi-good" if value >= 3 else "kpi-warn" if value >= 1 else "kpi-bad"
        elif metric == "roi":
            return "kpi-good" if value >= 2 else "kpi-warn" if value >= 0 else "kpi-bad"
        elif metric == "payback":
            return "kpi-good" if value and value <= 12 else "kpi-warn" if value else "kpi-bad"
        elif metric == "confidence":
            return "kpi-good" if value >= 70 else "kpi-warn" if value >= 40 else "kpi-bad"
        return "kpi-neutral"
    kpi_html = f"""
    {ENHANCED_KPI_CSS}
    <div class="kpi-grid">
        <div class="kpi-card kpi-neutral">
            <div class="kpi-label">💎 LTV</div>
            <div class="kpi-value">{data['ltv']:,.0f} ₽</div>
            <div class="kpi-change">Lifetime Value</div>
        </div>
        <div class="kpi-card {get_kpi_class('ltv_cac', data['ltv_cac'])}">
            <div class="kpi-label">📊 LTV/CAC</div>
            <div class="kpi-value">{data['ltv_cac']:.2f}</div>
            <div class="kpi-change">Эффективность привлечения</div>
        </div>
        <div class="kpi-card {get_kpi_class('payback', data['payback_month'])}">
            <div class="kpi-label">⏱️ Payback</div>
            <div class="kpi-value">{'Нет' if not data['payback_month'] else f"{data['payback_month']} мес."}</div>
            <div class="kpi-change">Срок окупаемости</div>
        </div>
        <div class="kpi-card {get_kpi_class('roi', data['roi'])}">
            <div class="kpi-label">📈 ROI</div>
            <div class="kpi-value">{data['roi']:.1%}</div>
            <div class="kpi-change">Возврат на инвестиции</div>
        </div>
        <div class="kpi-card kpi-neutral">
            <div class="kpi-label">⏳ Lifetime</div>
            <div class="kpi-value">{data['customer_lifetime']:.1f} мес.</div>
            <div class="kpi-change">Средний lifecycle</div>
        </div>
        <div class="kpi-card kpi-neutral">
            <div class="kpi-label">🔄 Retention</div>
            <div class="kpi-value">{data['annual_retention']:.1f}%</div>
            <div class="kpi-change">Годовое удержание</div>
        </div>
        <div class="kpi-card {get_kpi_class('confidence', data['confidence_score'])}">
            <div class="kpi-label">🎯 Confidence</div>
            <div class="kpi-value">{data['confidence_score']:.0f}%</div>
            <div class="kpi-change">Надежность модели</div>
        </div>
    </div>
    """
    return kpi_html

def create_insights_panel(insights):
    insights_html = '<div class="insights-panel"><h4>🧠 Инсайты и рекомендации</h4>'
    for insight in insights:
        insights_html += f"""
        <div class="insight-item">
            <div class="insight-icon">{insight['icon']}</div>
            <div class="insight-text">{insight['text']}</div>
        </div>
        """
    insights_html += '</div>'
    return insights_html

def generate_detailed_table(scenarios):
    all_data = []
    for scenario_name, data in scenarios.items():
        for i, month in enumerate(data['months'][:24]):
            all_data.append({
                'Сценарий': scenario_name,
                'Месяц': month,
                'Выживаемость (%)': round(data['survival_curve'][i] * 100, 2),
                'Месячный CF (₽)': round(data['monthly_cf'][i], 0),
                'Накопленный CF (₽)': round(data['cumulative_cf'][i], 0),
                'LTV до месяца (₽)': round(np.sum(data['monthly_cf'][:i+1]), 0)
            })
    df = pd.DataFrame(all_data)
    return df

def calculate_enhanced_ltv(avg_check, purchases_per_year, margin_pct, cac, 
                          monthly_churn_pct, discount_rate_pct, horizon_months, industry):
    params = {
        'avg_check': float(avg_check),
        'purchases_per_year': float(purchases_per_year), 
        'margin_pct': float(margin_pct),
        'cac': float(cac),
        'monthly_churn_pct': float(monthly_churn_pct),
        'discount_rate_pct': float(discount_rate_pct),
        'horizon_months': int(horizon_months)
    }
    warnings_list, errors = model.validate_inputs(params)
    if errors:
        error_msg = "<div style='color:#ff5f73;padding:16px;background:#2a1a1a;border-radius:8px;margin:8px 0'>"
        error_msg += "<h4>❌ Ошибки валидации:</h4><ul>"
        for error in errors:
            error_msg += f"<li>{error}</li>"
        error_msg += "</ul></div>"
        return error_msg, None, None, None, pd.DataFrame()
    scenarios = model.calculate_scenarios(params)
    base_data = scenarios["Базовый"]
    insights = model.generate_insights(base_data, params, industry)
    kpi_html = create_enhanced_kpi_cards(scenarios)
    if warnings_list:
        warning_msg = "<div class='warning-panel'><h5>⚠️ Предупреждения:</h5><ul>"
        for warning in warnings_list:
            warning_msg += f"<li>{warning}</li>"
        warning_msg += "</ul></div>"
        kpi_html = warning_msg + kpi_html
    insights_html = create_insights_panel(insights)
    kpi_html += insights_html
    scenarios_chart = create_scenarios_chart(scenarios)
    survival_chart = create_survival_chart(scenarios)
    sensitivity_chart = create_sensitivity_chart(params)
    detailed_table = generate_detailed_table(scenarios)
    return kpi_html, scenarios_chart, survival_chart, sensitivity_chart, detailed_table

def generate_recommendations(avg_check, cac, margin_pct, monthly_churn_pct):
    recs = []
    if avg_check < cac:
        recs.append("🎯 **Увеличить средний чек**: внедрить апсейл и кросс-сейл")
    if margin_pct < 30:
        recs.append("💰 **Оптимизировать маржинальность**: пересмотреть ценообразование")
    if monthly_churn_pct > 10:
        recs.append("🔒 **Улучшить удержание**: программы лояльности, customer success")
    if cac > avg_check * 0.8:
        recs.append("📢 **Оптимизировать маркетинг**: улучшить targeting и конверсию")
    if not recs:
        recs.append("✨ **Отличные показатели!** Продолжайте мониторинг и A/B тестирование")
    html = "<div style='padding:12px'><ul>"
    for rec in recs:
        html += f"<li style='margin:8px 0;color:#222'>{rec}</li>"  # Тёмный текст рекомендаций
    html += "</ul></div>"
    return html

with gr.Blocks(
    css="""
    .gradio-container { max-width: 1400px !important; }
    .panel { background: #1a1a1a; border-radius: 12px; padding: 16px; margin: 8px 0; }
    """,
    theme=gr.themes.Soft(primary_hue="amber", secondary_hue="gray"),
    title="ULTIMA — Advanced LTV Analytics"
) as demo:
    gr.HTML(ENHANCED_HEADER_HTML)
    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("## 🎛️ **Настройки модели**")
            with gr.Accordion("💰 Финансовые параметры", open=True):
                avg_check = gr.Number(
                    label="💵 Средний чек (₽)", value=20000, precision=0,
                    info="Средняя сумма покупки одного клиента"
                )
                purchases_per_year = gr.Number(
                    label="🛒 Покупок в год", value=2.5, precision=1,
                    info="Частота покупок клиента в год"
                )
                margin_pct = gr.Slider(
                    label="📊 Маржинальность (%)", value=50, minimum=0, maximum=100, step=1,
                    info="Процент прибыли с каждой продажи"
                )
                cac = gr.Number(
                    label="💸 CAC (₽)", value=15000, precision=0,
                    info="Стоимость привлечения одного клиента"
                )
            with gr.Accordion("📈 Параметры удержания", open=True):
                monthly_churn_pct = gr.Slider(
                    label="📉 Месячный отток (%)", value=8, minimum=0, maximum=50, step=0.5,
                    info="Процент клиентов, покидающих каждый месяц"
                )
                discount_rate_pct = gr.Slider(
                    label="💹 Ставка дисконтирования (%)", value=12, minimum=0, maximum=30, step=1,
                    info="Годовая ставка для дисконтирования будущих денежных потоков"
                )
            with gr.Accordion("⚙️ Дополнительные настройки", open=False):
                horizon_months = gr.Dropdown(
                    label="📅 Горизонт планирования", 
                    choices=[12, 24, 36, 48, 60], value=36,
                    info="Период расчета LTV в месяцах"
                )
                industry = gr.Dropdown(
                    label="🏢 Отрасль (для бенчмарков)",
                    choices=["SaaS", "E-commerce", "Marketplace", "Fintech"], value="SaaS",
                    info="Отрасль для сравнения с типичными показателями"
                )
            calculate_btn = gr.Button("🚀 Рассчитать LTV", variant="primary", size="lg")
        with gr.Column(scale=2):
            gr.Markdown("## 📊 **Аналитика и результаты**")
            kpi_output = gr.HTML()
            with gr.Tabs():
                with gr.TabItem("📈 Cash Flow Analysis"):
                    scenarios_plot = gr.Plot()
                with gr.TabItem("👥 Customer Retention"):
                    survival_plot = gr.Plot()
                with gr.TabItem("🎯 Sensitivity Analysis"):
                    sensitivity_plot = gr.Plot()
                with gr.TabItem("📋 Detailed Data"):
                    detailed_table = gr.Dataframe(
                        label="Детализированные данные по месяцам и сценариям",
                        wrap=True
                    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🔍 **Дополнительная аналитика**")
            with gr.Accordion("💡 Рекомендации по улучшению", open=False):
                recommendations_html = gr.HTML()
            with gr.Accordion("📚 Методология расчета", open=False):
                gr.Markdown("""
                ### Формула расчета LTV

                **LTV = Σ(Monthly Margin × Survival Rate × Discount Factor)**

                Где:
                - **Monthly Margin** = (Средний чек × Покупок в год / 12) × Маржинальность
                - **Survival Rate** = (1 - Monthly Churn)^(месяц-1)
                - **Discount Factor** = (1 + Discount Rate)^(-месяц/12)

                ### Ключевые метрики

                - **LTV/CAC ≥ 3.0** — минимальный порог эффективности
                - **Payback ≤ 12 мес** — желаемый срок окупаемости для большинства бизнесов
                - **Monthly Churn < 10%** — здоровый уровень оттока
                - **Customer Lifetime** = 1 / Monthly Churn Rate

                ### Сценарное моделирование

                - **Пессимистичный**: +50% оттока, -20% маржи, -10% среднего чека
                - **Оптимистичный**: -30% оттока, +20% маржи, +10% среднего чека
                """)

    gr.HTML(f"""
    <div style="margin-top:24px;padding:20px;background:{ULTIMA_DARK};border-radius:12px;text-align:center">
        <img src="https://cdn-uploads.huggingface.co/production/uploads/68a5d644d41e00d772823934/bmu2UTnqh39vYO0wRV718.png" style="width:40px;height:40px;margin-right:12px;vertical-align:middle;" />
        <span style="color:{ULTIMA_GOLD};font-weight:600;font-size:18px;vertical-align:middle;">
            🚀 ULTIMA Advanced LTV Analytics v2.0
        </span>
        <div style="color:#666;font-size:12px;margin-top:8px">
            Разработано с использованием передовых методов финансового моделирования<br>
            Поддержка сценарного анализа • Отраслевые бенчмарки • Интерактивная визуализация
        </div>
    </div>
    """)

    calculate_btn.click(
        fn=calculate_enhanced_ltv,
        inputs=[
            avg_check, purchases_per_year, margin_pct, cac, 
            monthly_churn_pct, discount_rate_pct, horizon_months, industry
        ],
        outputs=[
            kpi_output, scenarios_plot, survival_plot, 
            sensitivity_plot, detailed_table
        ]
    )

    calculate_btn.click(
        fn=generate_recommendations,
        inputs=[avg_check, cac, margin_pct, monthly_churn_pct],
        outputs=[recommendations_html]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        show_error=True,
        favicon_path=None,
        ssl_verify=False
    )
