async function analyzeEmotion() {
    const text = document.getElementById('userInput').value;
    const btn = document.getElementById('btnAnalyze');
    const resultDiv = document.getElementById('resultContainer');
    const confidenceBar = document.getElementById('confidenceBar');

    if (!text.trim()) {
        alert("Please enter some text to analyze.");
        return;
    }

    // Loading State
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing AI...';
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        // Update UI
        document.getElementById('emotionLabel').innerText = data.emotion.toUpperCase();
        document.getElementById('confidenceLabel').innerText = data.confidence;
        
        // Update Progress Bar
        const confValue = parseFloat(data.confidence);
        confidenceBar.style.width = confValue + "%";

        resultDiv.classList.remove('d-none');
        resultDiv.classList.add('animate-up');

    } catch (error) {
        console.error("Error:", error);
    } finally {
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-2"></i>Analyze Sentiment';
        btn.disabled = false;
    }
}
