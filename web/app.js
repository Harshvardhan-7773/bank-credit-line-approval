/**
 * ApexBank India Credit Underwriting Client Application (INR / ₹).
 * Powered by XGBoost and Logistic Regression.
 */

let currentPresets = [];
let metadata = null;
let rocChartInstance = null;
let featureChartInstance = null;
let selectedModel = 'xgboost';

// Format Indian Rupee currency (e.g. ₹14,50,000)
function formatINR(val) {
  if (val === undefined || val === null || isNaN(val)) return '₹0';
  const num = Math.round(val);
  const s = num.toString();
  if (s.length <= 3) return '₹' + s;
  const lastThree = s.substring(s.length - 3);
  let otherNumbers = s.substring(0, s.length - 3);
  const chunks = [];
  while (otherNumbers.length > 2) {
    chunks.unshift(otherNumbers.substring(otherNumbers.length - 2));
    otherNumbers = otherNumbers.substring(0, otherNumbers.length - 2);
  }
  if (otherNumbers.length > 0) chunks.unshift(otherNumbers);
  return '₹' + chunks.join(',') + ',' + lastThree;
}

// Model Selection Toggle
function selectModel(modelName) {
  selectedModel = modelName;
  const btnXgb = document.getElementById('btn-model-xgb');
  const btnLr = document.getElementById('btn-model-lr');

  if (modelName === 'xgboost') {
    btnXgb.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-orange-600 text-white shadow transition";
    btnLr.className = "px-3 py-1 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition";
  } else {
    btnLr.className = "px-3 py-1 rounded-lg text-xs font-semibold bg-orange-600 text-white shadow transition";
    btnXgb.className = "px-3 py-1 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition";
  }

  submitApplication();
}

// Tab Switching
function switchTab(tabName) {
  document.querySelectorAll('.tab-view').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('bg-orange-600', 'text-white', 'shadow');
    btn.classList.add('text-slate-400');
  });

  const targetView = document.getElementById(`view-${tabName}`);
  const targetBtn = document.getElementById(`tab-${tabName}`);

  if (targetView && targetBtn) {
    targetView.classList.remove('hidden');
    targetBtn.classList.remove('text-slate-400');
    targetBtn.classList.add('bg-orange-600', 'text-white', 'shadow');
  }

  if (tabName === 'models' && metadata && !rocChartInstance) {
    renderCharts(metadata);
  }
}

// Fetch Initial Presets
async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    currentPresets = await res.json();
    const container = document.getElementById('preset-buttons');
    container.innerHTML = '';

    currentPresets.forEach((preset) => {
      const btn = document.createElement('button');
      btn.className = "text-left p-2.5 rounded-xl border border-slate-800 bg-slate-950/70 hover:border-orange-500/50 hover:bg-slate-900 transition flex flex-col justify-between";
      btn.onclick = () => applyPreset(preset);
      btn.innerHTML = `
        <span class="text-xs font-bold text-white">${preset.name.split(' (')[0]}</span>
        <span class="text-[10px] text-orange-400 font-mono mt-0.5">${preset.name.includes('Super') ? 'Super Prime' : preset.name.includes('Builder') ? 'Prime' : preset.name.includes('Review') ? 'Review' : 'Decline'}</span>
      `;
      container.appendChild(btn);
    });
  } catch (err) {
    console.error("Error loading presets:", err);
  }
}

// Apply Preset Data to Form
function applyPreset(preset) {
  const d = preset.data;
  document.getElementById('display-app-id').innerText = `ID: ${d.applicant_id}`;
  document.getElementById('annual_income').value = d.annual_income;
  document.getElementById('income-hint').innerText = `${formatINR(d.annual_income)} (${(d.annual_income / 100000).toFixed(1)} Lakhs / yr)`;
  document.getElementById('credit_score').value = d.credit_score;
  document.getElementById('debt_to_income').value = d.debt_to_income;
  document.getElementById('val-dti').innerText = (d.debt_to_income * 100).toFixed(1) + '%';
  document.getElementById('revolving_utilization').value = d.revolving_utilization;
  document.getElementById('val-util').innerText = (d.revolving_utilization * 100).toFixed(1) + '%';
  document.getElementById('revolving_balance').value = d.revolving_balance;
  document.getElementById('credit_history_years').value = d.credit_history_years;
  document.getElementById('age').value = d.age;
  document.getElementById('employment_years').value = d.employment_years;
  document.getElementById('delinquencies_2yrs').value = d.delinquencies_2yrs;
  document.getElementById('num_derogatory_records').value = d.num_derogatory_records;
  document.getElementById('num_credit_inquiries_6m').value = d.num_credit_inquiries_6m;
  document.getElementById('home_ownership').value = d.home_ownership;
  document.getElementById('loan_purpose').value = d.loan_purpose;
  document.getElementById('bankruptcy_history').value = d.bankruptcy_history;

  submitApplication();
}

// Submit Application for Underwriting
async function submitApplication() {
  const annualIncome = parseFloat(document.getElementById('annual_income').value);
  document.getElementById('income-hint').innerText = `${formatINR(annualIncome)} (${(annualIncome / 100000).toFixed(1)} Lakhs / yr)`;

  const payload = {
    applicant_id: document.getElementById('display-app-id').innerText.replace('ID: ', ''),
    model_choice: selectedModel,
    annual_income: annualIncome,
    credit_score: parseInt(document.getElementById('credit_score').value), // CIBIL
    debt_to_income: parseFloat(document.getElementById('debt_to_income').value),
    revolving_utilization: parseFloat(document.getElementById('revolving_utilization').value),
    revolving_balance: parseFloat(document.getElementById('revolving_balance').value),
    credit_history_years: parseFloat(document.getElementById('credit_history_years').value),
    age: parseInt(document.getElementById('age').value),
    employment_years: parseFloat(document.getElementById('employment_years').value),
    delinquencies_2yrs: parseInt(document.getElementById('delinquencies_2yrs').value),
    num_derogatory_records: parseInt(document.getElementById('num_derogatory_records').value),
    num_credit_inquiries_6m: parseInt(document.getElementById('num_credit_inquiries_6m').value),
    home_ownership: document.getElementById('home_ownership').value,
    loan_purpose: document.getElementById('loan_purpose').value,
    bankruptcy_history: parseInt(document.getElementById('bankruptcy_history').value)
  };

  try {
    const res = await fetch('/api/underwrite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderDecision(data);
  } catch (err) {
    console.error("Underwriting failed:", err);
  }
}

// Render Decision Hero and Explainability
function renderDecision(data) {
  const card = document.getElementById('decision-card');
  const decElem = document.getElementById('display-decision');
  const tierElem = document.getElementById('display-risk-tier');
  const pdElem = document.getElementById('display-pd');
  const confElem = document.getElementById('display-confidence');
  const lineElem = document.getElementById('display-credit-line');
  const aprElem = document.getElementById('display-apr');
  const modelBadge = document.getElementById('display-model-badge');

  modelBadge.innerText = data.model_used;
  card.classList.remove('glow-approved', 'glow-review', 'glow-declined');

  if (data.decision === 'APPROVED') {
    card.classList.add('glow-approved');
    decElem.className = "text-3xl font-extrabold tracking-tight mt-1 text-emerald-400 flex items-center";
    decElem.innerHTML = `<i class="fa-solid fa-circle-check mr-2"></i> APPROVED`;
  } else if (data.decision === 'MANUAL_REVIEW') {
    card.classList.add('glow-review');
    decElem.className = "text-3xl font-extrabold tracking-tight mt-1 text-amber-400 flex items-center";
    decElem.innerHTML = `<i class="fa-solid fa-triangle-exclamation mr-2"></i> MANUAL REVIEW`;
  } else {
    card.classList.add('glow-declined');
    decElem.className = "text-3xl font-extrabold tracking-tight mt-1 text-rose-500 flex items-center";
    decElem.innerHTML = `<i class="fa-solid fa-circle-xmark mr-2"></i> DECLINED`;
  }

  tierElem.innerText = `Risk Tier: ${data.risk_tier}`;
  pdElem.innerText = `${(data.probability_of_default * 100).toFixed(1)}%`;
  confElem.innerText = `Confidence: ${(data.approval_confidence * 100).toFixed(1)}%`;
  lineElem.innerText = data.formatted_credit_line;
  aprElem.innerText = data.suggested_pricing_apr;

  // Snapshot
  document.getElementById('snap-disposable').innerText = data.financial_summary.monthly_disposable;
  document.getElementById('snap-dti').innerText = data.financial_summary.foir_dti_pct;
  document.getElementById('snap-fico').innerText = data.financial_summary.cibil_score;

  // Model comparison display
  document.getElementById('cmp-xgb').innerText = `${(data.model_comparison.xgboost_approval_prob * 100).toFixed(1)}%`;
  document.getElementById('cmp-lr').innerText = `${(data.model_comparison.logistic_regression_approval_prob * 100).toFixed(1)}%`;

  // Explainability
  const explainHeader = document.getElementById('explain-header');
  const explainBadge = document.getElementById('explain-badge');
  const explainContent = document.getElementById('explain-content');
  explainContent.innerHTML = '';

  if (data.decision === 'DECLINED') {
    explainHeader.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-rose-400 mr-2"></i> RBI Fair Practices Rejection Disclosure`;
    explainBadge.className = "text-xs px-2.5 py-0.5 rounded-full font-mono bg-rose-950 text-rose-300 border border-rose-800";
    explainBadge.innerText = "Mandatory Disclosure";

    if (data.hard_knockout_triggered) {
      const koBox = document.createElement('div');
      koBox.className = "p-3 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-200 text-xs";
      koBox.innerHTML = `<strong>RBI Regulatory Knockout:</strong> ${data.hard_knockout_reasons.join(', ')}`;
      explainContent.appendChild(koBox);
    }

    data.adverse_action_reasons.forEach(r => {
      const item = document.createElement('div');
      item.className = "p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 flex items-start space-x-3";
      item.innerHTML = `
        <span class="text-xs font-mono font-bold px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-900">${r.code}</span>
        <div>
          <h5 class="text-xs font-bold text-white">${r.title}</h5>
          <p class="text-[11px] text-slate-400 mt-0.5">${r.detail}</p>
        </div>
      `;
      explainContent.appendChild(item);
    });

  } else {
    explainHeader.innerHTML = `<i class="fa-solid fa-shield-halved text-emerald-400 mr-2"></i> Key Underwriting Strengths`;
    explainBadge.className = "text-xs px-2.5 py-0.5 rounded-full font-mono bg-emerald-950 text-emerald-300 border border-emerald-800";
    explainBadge.innerText = "Positive Factors";

    data.key_strengths.forEach(s => {
      const item = document.createElement('div');
      item.className = "p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 flex items-start space-x-3";
      item.innerHTML = `
        <i class="fa-solid fa-check text-emerald-400 mt-1"></i>
        <div>
          <h5 class="text-xs font-bold text-white">${s.title}</h5>
          <p class="text-[11px] text-slate-400 mt-0.5">${s.detail}</p>
        </div>
      `;
      explainContent.appendChild(item);
    });
  }
}

// Fetch Metadata & Render Benchmark Table
async function loadMetadata() {
  try {
    const res = await fetch('/api/metadata');
    metadata = await res.json();

    const tbody = document.getElementById('model-benchmark-body');
    tbody.innerHTML = '';
    const benchmarks = metadata.model_benchmarks;
    const champion = metadata.champion_classifier;

    for (const [name, metrics] of Object.entries(benchmarks)) {
      const isChamp = name === champion;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="p-3 text-white font-bold">${name}</td>
        <td class="p-3">${(metrics.accuracy * 100).toFixed(2)}%</td>
        <td class="p-3">${(metrics.precision * 100).toFixed(2)}%</td>
        <td class="p-3">${(metrics.recall * 100).toFixed(2)}%</td>
        <td class="p-3">${(metrics.f1_score * 100).toFixed(2)}%</td>
        <td class="p-3 text-orange-400 font-bold">${metrics.roc_auc.toFixed(4)}</td>
        <td class="p-3">
          ${isChamp ? '<span class="px-2 py-0.5 rounded bg-orange-950 text-orange-300 border border-orange-800 text-[10px] font-bold">CHAMPION</span>' : '<span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px]">BASEL BENCHMARK</span>'}
        </td>
      `;
      tbody.appendChild(tr);
    }

    document.getElementById('roc-auc-badge').innerText = `ROC-AUC: ${metadata.classifier_metrics.roc_auc.toFixed(4)}`;
  } catch (err) {
    console.error("Error loading metadata:", err);
  }
}

// Render Chart.js ROC Curve and XGBoost Feature Importances
function renderCharts(data) {
  const rocCtx = document.getElementById('rocChart').getContext('2d');
  const rocPoints = data.classifier_metrics.roc_curve;
  
  rocChartInstance = new Chart(rocCtx, {
    type: 'line',
    data: {
      labels: rocPoints.map(p => p.fpr),
      datasets: [
        {
          label: 'XGBoost Classifier ROC Curve',
          data: rocPoints.map(p => p.tpr),
          borderColor: '#ea580c',
          backgroundColor: 'rgba(234, 88, 12, 0.1)',
          fill: true,
          tension: 0.2,
          pointRadius: 0
        },
        {
          label: 'Random Guess (AUC 0.50)',
          data: rocPoints.map(p => p.fpr),
          borderColor: '#475569',
          borderDash: [5, 5],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'False Positive Rate (FPR)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { title: { display: true, text: 'True Positive Rate (TPR)', color: '#94a3b8' }, grid: { color: '#1e293b' }, ticks: { color: '#64748b' } }
      },
      plugins: {
        legend: { labels: { color: '#cbd5e1' } }
      }
    }
  });

  const featCtx = document.getElementById('featureChart').getContext('2d');
  const feats = data.top_feature_importances.slice(0, 8);

  featureChartInstance = new Chart(featCtx, {
    type: 'bar',
    data: {
      labels: feats.map(f => f.feature.replace('num__', '').replace('cat__', '')),
      datasets: [{
        label: 'XGBoost Feature Importance (Gain)',
        data: feats.map(f => f.importance),
        backgroundColor: '#f97316',
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Run Sample Batch Underwriting in INR
async function runSampleBatch() {
  const applicants = [];
  const baseApps = currentPresets.map(p => p.data);

  for (let i = 0; i < 25; i++) {
    const template = baseApps[i % baseApps.length];
    const jitter = (Math.random() - 0.5) * 0.25;
    applicants.push({
      ...template,
      applicant_id: `IN-COHORT-${1000 + i}`,
      annual_income: Math.round(template.annual_income * (1 + jitter)),
      credit_score: Math.max(300, Math.min(900, Math.round(template.credit_score + (Math.random() - 0.5) * 45))),
      debt_to_income: Math.max(0.08, Math.min(0.68, template.debt_to_income + (Math.random() - 0.5) * 0.08))
    });
  }

  try {
    const res = await fetch('/api/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ applicants, model_choice: selectedModel })
    });
    const data = await res.json();
    
    document.getElementById('batch-total').innerText = data.summary.total_applications;
    document.getElementById('batch-approval-rate').innerText = `${data.summary.approval_rate_pct}%`;
    document.getElementById('batch-total-capital').innerText = data.summary.formatted_total_capital;
    document.getElementById('batch-avg-line').innerText = data.summary.formatted_avg_line;

    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';

    data.applications.forEach(app => {
      const tr = document.createElement('tr');
      const isApp = app.decision === 'APPROVED';
      const isRev = app.decision === 'MANUAL_REVIEW';
      const decBadge = isApp 
        ? '<span class="text-emerald-400 font-bold">APPROVED</span>' 
        : isRev 
        ? '<span class="text-amber-400 font-bold">REVIEW</span>' 
        : '<span class="text-rose-500 font-bold">DECLINED</span>';

      tr.innerHTML = `
        <td class="p-3 text-white">${app.applicant_id}</td>
        <td class="p-3">${app.financial_summary.annual_income}</td>
        <td class="p-3">${app.financial_summary.cibil_score}</td>
        <td class="p-3">${app.financial_summary.foir_dti_pct}</td>
        <td class="p-3 text-orange-400">${(app.probability_of_default * 100).toFixed(1)}%</td>
        <td class="p-3">${decBadge}</td>
        <td class="p-3 text-white font-bold">${app.formatted_credit_line}</td>
        <td class="p-3 text-slate-400">${app.risk_tier.split(' ')[0]}</td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Batch run failed:", err);
  }
}

// Initialization on DOM Load
window.addEventListener('DOMContentLoaded', async () => {
  await loadPresets();
  await loadMetadata();
  if (currentPresets.length > 0) {
    applyPreset(currentPresets[0]);
  }
});
