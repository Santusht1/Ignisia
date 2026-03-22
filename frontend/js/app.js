// Global Configuration
Chart.defaults.color = '#8b949e';
Chart.defaults.font.family = "'Inter', sans-serif";

// State
let salesChartInstance = null;
let importanceChartInstance = null;

// DOM Elements
const fileInput = document.getElementById('csv-file');
const fileNameDisplay = document.getElementById('file-name');
const uploadForm = document.getElementById('upload-form');
const analyzeBtn = document.getElementById('analyze-btn');
const uploadStatus = document.getElementById('upload-status');
const dashboardContent = document.getElementById('dashboard-content');
const modelInfo = document.getElementById('model-info');
const insightsContainer = document.getElementById('insights-container');

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');

// Event Listeners
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        fileNameDisplay.textContent = e.target.files[0].name;
    } else {
        fileNameDisplay.textContent = 'Choose CSV file';
    }
});

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    uploadStatus.textContent = '';
    uploadStatus.style.color = 'inherit';

    try {
        const response = await fetch('https://ignisia-0xsc.onrender.com/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            uploadStatus.textContent = 'Data processed successfully! \u2713';
            uploadStatus.style.color = 'var(--positive)';
            populateDashboard(data);
        } else {
            throw new Error(data.detail || 'Upload failed');
        }
    } catch (error) {
        uploadStatus.textContent = 'Error: ' + error.message;
        uploadStatus.style.color = 'var(--danger)';
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Generate Insights';
    }
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = chatInput.value.trim();
    if (!msg) return;

    appendMessage('user', msg);
    chatInput.value = '';

    try {
        const response = await fetch('https://ignisia-0xsc.onrender.com/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });
        const data = await response.json();
        appendMessage('assistant', data.response);
    } catch (e) {
        appendMessage('assistant', 'Sorry, I encountered an error communicating with the server.');
    }
});

// UI Functions
function appendMessage(role, text) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    const p = document.createElement('p');
    p.textContent = text;
    div.appendChild(p);
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function populateDashboard(data) {
    // Unlock UI
    dashboardContent.style.opacity = '1';
    dashboardContent.style.pointerEvents = 'auto';

    const results = data.results;
    
    // Model Info Updates
    modelInfo.style.display = 'block';
    document.getElementById('model-name').textContent = results.best_model;
    document.getElementById('model-r2').textContent = results.metrics.R2.toFixed(3);
    document.getElementById('model-mae').textContent = results.metrics.MAE.toFixed(2);

    // Insights Updates
    renderInsights(data.insights);

    // Charts
    renderSalesChart(results.historical, results.predictions);
    renderImportanceChart(results.feature_importance);
}

function renderInsights(insights) {
    insightsContainer.innerHTML = '';
    if (!insights || insights.length === 0) {
        insightsContainer.innerHTML = '<div class="empty-state">No significant insights generated.</div>';
        return;
    }

    insights.forEach(ins => {
        const div = document.createElement('div');
        div.className = `insight-card ${ins.type || 'neutral'}`;
        div.innerHTML = `
            <h4>${ins.title}</h4>
            <p>${ins.message}</p>
        `;
        insightsContainer.appendChild(div);
    });
}

function renderSalesChart(historical, predictions) {
    const ctx = document.getElementById('salesChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (salesChartInstance) {
        salesChartInstance.destroy();
    }

    const histLabels = historical.map(d => d.Date);
    const histData = historical.map(d => d.Sales);

    const predLabels = predictions.map(d => d.Date);
    const predData = predictions.map(d => d.Predicted_Sales);

    // Combine labels uniquely to create a continuous x-axis
    const labels = [...histLabels, ...predLabels];
    
    // Pad predData with nulls for historical period so lines connect correctly
    const paddedPredData = new Array(historical.length - 1).fill(null);
    paddedPredData.push(historical[historical.length - 1].Sales); // connect
    paddedPredData.push(...predData);

    const fullHistData = [...histData, ...new Array(predictions.length).fill(null)];

    salesChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical Sales',
                    data: fullHistData,
                    borderColor: 'rgba(88, 166, 255, 1)',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 5
                },
                {
                    label: 'AI Forecast',
                    data: paddedPredData,
                    borderColor: 'rgba(63, 185, 80, 1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.3,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } },
                tooltip: { backgroundColor: 'rgba(22, 27, 34, 0.9)' }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(48, 54, 61, 0.2)' },
                    ticks: { maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(48, 54, 61, 0.2)' },
                    beginAtZero: true
                }
            }
        }
    });
}

function renderImportanceChart(importanceDict) {
    const ctx = document.getElementById('importanceChart').getContext('2d');
    
    if (importanceChartInstance) {
        importanceChartInstance.destroy();
    }

    if (!importanceDict || Object.keys(importanceDict).length === 0) {
        return;
    }

    const labels = Object.keys(importanceDict);
    const data = Object.values(importanceDict);

    // Sort by importance
    const sortedIndices = data.map((d, i) => i).sort((a, b) => data[b] - data[a]);
    const sortedLabels = sortedIndices.map(i => {
        // Humanize labels
        const lbl = labels[i];
        if (lbl === 'day_of_week') return 'Day of Week';
        if (lbl === 'is_weekend') return 'Is Weekend';
        if (lbl === 'lag_1') return 'Previous Day Sales';
        if (lbl === 'lag_7') return 'Previous Week Sales';
        return lbl;
    }).slice(0, 5); // top 5
    const sortedData = sortedIndices.map(i => data[i]).slice(0, 5);

    importanceChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedLabels,
            datasets: [{
                label: 'Relative Impact',
                data: sortedData,
                backgroundColor: 'rgba(139, 148, 158, 0.5)',
                borderColor: 'rgba(139, 148, 158, 1)',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y', // Horizontal bar chart
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false }, // Hide x axis as relative scale
                y: { grid: { display: false } }
            }
        }
    });
}
