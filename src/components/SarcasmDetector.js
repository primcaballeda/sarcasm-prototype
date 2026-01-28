import React, { useState } from 'react';
import './SarcasmDetector.css';

const SarcasmDetector = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  // Simple rule-based sarcasm detection (prototype)
  const detectSarcasm = (inputText) => {
    const lowerText = inputText.toLowerCase();
    
    // Sarcasm indicators
    const sarcasmKeywords = [
      'yeah right',
      'sure',
      'obviously',
      'oh great',
      'fantastic',
      'wonderful',
      'brilliant',
      'perfect',
      'just what i needed',
      'oh wow',
      'shocking'
    ];
    
    const hasExclamation = (inputText.match(/!/g) || []).length >= 2;
    const hasQuotes = inputText.includes('"') || inputText.includes("'");
    const hasEllipsis = inputText.includes('...');
    
    let sarcasmScore = 0;
    let indicators = [];
    
    // Check for sarcasm keywords
    sarcasmKeywords.forEach(keyword => {
      if (lowerText.includes(keyword)) {
        sarcasmScore += 30;
        indicators.push(`Contains phrase: "${keyword}"`);
      }
    });
    
    // Check punctuation patterns
    if (hasExclamation) {
      sarcasmScore += 15;
      indicators.push('Multiple exclamation marks');
    }
    
    if (hasQuotes) {
      sarcasmScore += 10;
      indicators.push('Contains quotes (air quotes?)');
    }
    
    if (hasEllipsis) {
      sarcasmScore += 10;
      indicators.push('Contains ellipsis (trailing off)');
    }
    
    // Check for ALL CAPS words
    const words = inputText.split(' ');
    const capsWords = words.filter(word => 
      word.length > 2 && word === word.toUpperCase() && /[A-Z]/.test(word)
    );
    
    if (capsWords.length > 0) {
      sarcasmScore += 15;
      indicators.push(`Emphasis with caps: ${capsWords.join(', ')}`);
    }
    
    // Determine result
    const isSarcastic = sarcasmScore > 25;
    const confidence = Math.min(sarcasmScore, 100);
    
    return {
      isSarcastic,
      confidence,
      indicators: indicators.length > 0 ? indicators : ['No strong sarcasm indicators found']
    };
  };

  const handleAnalyze = () => {
    if (!text.trim()) {
      return;
    }

    setAnalyzing(true);
    
    // Simulate processing delay
    setTimeout(() => {
      const detection = detectSarcasm(text);
      setResult(detection);
      setAnalyzing(false);
    }, 800);
  };

  const handleReset = () => {
    setText('');
    setResult(null);
  };

  return (
    <div className="sarcasm-detector">
      <div className="header">
        <h1>Sarcasm Detector</h1>
        <p className="subtitle">Is it sarcasm or not? Let's find out!</p>
      </div>

      <div className="input-section">
        <textarea
          className="text-input"
          placeholder="Type or paste text to analyze... (e.g., 'Oh great, another meeting!')"
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={6}
        />
        
        <div className="button-group">
          <button 
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={!text.trim() || analyzing}
          >
            {analyzing ? 'Analyzing...' : 'Detect Sarcasm'}
          </button>
          
          {text && (
            <button 
              className="reset-btn"
              onClick={handleReset}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {result && (
        <div className={`result-section ${result.isSarcastic ? 'sarcastic' : 'not-sarcastic'}`}>
          <div className="result-header">
            <h2>
              {result.isSarcastic ? 'SARCASM DETECTED!' : 'Not Sarcastic'}
            </h2>
          </div>
          
          <div className="confidence-bar">
            <div className="confidence-label">
              Confidence: {result.confidence}%
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${result.confidence}%` }}
              />
            </div>
          </div>

          <div className="indicators">
            <h3>Indicators:</h3>
            <ul>
              {result.indicators.map((indicator, index) => (
                <li key={index}>{indicator}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="examples">
        <h3>Try these examples:</h3>
        <div className="example-buttons">
          <button 
            className="example-btn"
            onClick={() => setText("Oh great, another Monday morning meeting!")}
          >
            Example 1
          </button>
          <button 
            className="example-btn"
            onClick={() => setText("Yeah right, like that's ever going to happen...")}
          >
            Example 2
          </button>
          <button 
            className="example-btn"
            onClick={() => setText("I love working on weekends!")}
          >
            Example 3
          </button>
          <button 
            className="example-btn"
            onClick={() => setText("Thank you for your help today.")}
          >
            Example 4
          </button>
        </div>
      </div>

      <div className="disclaimer">
        <p>
          <strong>Note:</strong> This is a prototype using simple rule-based detection. 
          A production system would use machine learning models for better accuracy.
        </p>
      </div>
    </div>
  );
};

export default SarcasmDetector;
