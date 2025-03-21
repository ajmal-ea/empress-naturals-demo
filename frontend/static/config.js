// Configuration for Express Analytics Chatbot
const EA_CHATBOT_CONFIG = {
    // API endpoint URL - Automatically detect environment
    apiBaseUrl: detectApiUrl(),
    
    // Chatbot appearance
    chatbotTitle: 'Empress Naturals Skincare Assistant',
    logoUrl: 'https://eco-cdn.iqpc.com/eco/images/partners/bERctvqEJdQH4Rnc0fMEhfgPlrjunOLsPPFvbKtR.png',
    
    // Welcome message
    welcomeMessage: `👋 Welcome to Empress Naturals! I'm your skincare assistant, ready to help you with:
    <ul>
        <li>Natural skincare for perimenopause and menopause</li>
        <li>Information about our organic ingredients</li>
        <li>Product recommendations for your skin concerns</li>
        <li>Questions about our Empress Serums and body oils</li>
    </ul>
    How can I assist with your royal skincare ritual today?`,
    
    // Meeting scheduler
    meetingSchedulerUrl: 'https://calendly.com/empressnaturals/consultation',
    
    // Branding
    poweredByText: 'Powered by Express Analytics',
    poweredByLink: 'https://www.expressanalytics.com'
};

// Function to detect the appropriate API URL based on environment
function detectApiUrl() {
    // Check if we're running in Docker by looking at the hostname
    const isDocker = window.location.hostname === 'localhost' && 
                     (window.location.port === '4000' || window.location.port === '');
    
    if (isDocker) {
        // When running in Docker, use the service name from docker-compose
        return 'https://empress-naturals-ea-bot-backend-production.up.railway.app';
    } else if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        // Local development outside Docker
        return 'https://empress-naturals-ea-bot-backend-production.up.railway.app';
    } else {
        // Production environment - replace with your actual production API URL
        return 'https://empress-naturals-ea-bot-backend-production.up.railway.app';
    }
} 