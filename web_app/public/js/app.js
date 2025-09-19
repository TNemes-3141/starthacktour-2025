// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Web app loaded successfully!');
    
    // Get references to DOM elements
    const apiTestBtn = document.getElementById('api-test-btn');
    const apiResult = document.getElementById('api-result');
    
    // API test button click handler
    apiTestBtn.addEventListener('click', async function() {
        try {
            // Show loading state
            apiTestBtn.textContent = 'Testing...';
            apiTestBtn.disabled = true;
            
            // Make API call
            const response = await fetch('/api/hello');
            const data = await response.json();
            
            // Show success result
            apiResult.className = 'show';
            apiResult.innerHTML = `
                <h4>✅ API Test Successful!</h4>
                <p><strong>Response:</strong> ${data.message}</p>
                <p><strong>Status:</strong> ${response.status}</p>
            `;
            
        } catch (error) {
            // Show error result
            apiResult.className = 'show error';
            apiResult.innerHTML = `
                <h4>❌ API Test Failed</h4>
                <p><strong>Error:</strong> ${error.message}</p>
            `;
        } finally {
            // Reset button state
            apiTestBtn.textContent = 'Test API';
            apiTestBtn.disabled = false;
        }
    });
    
    // Add some interactivity to feature cards
    const featureCards = document.querySelectorAll('.feature');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
});