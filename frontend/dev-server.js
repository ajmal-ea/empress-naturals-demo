const express = require('express');
const cors = require('cors');
const path = require('path');
const app = express();
const port = process.env.PORT || 10000;

// Enable CORS for all routes
app.use(cors());

// Serve static files from the 'static' directory
// UI modifications:
// 1. Removed chat history button
// 2. Disabled submit button in contact form
// 3. Updated color scheme to match empressnaturals.co (gold/bronze theme)
// 4. Updated branding and content to match Empress Naturals
// 5. Changed welcome message to focus on skincare for perimenopause and menopause
// 6. Added proper markdown parsing for headers and improved formatting of responses
// 7. Fixed handling of bullet points (asterisks) that appear in the middle of lines
// 8. Added support for numbered/ordered lists (1., 2., etc.) and h4 headers
// 9. Completely rewrote markdown parser to properly handle paragraph spacing, multi-line lists, and bold text
app.use(express.static(path.join(__dirname, 'static')));

// Serve the test-embed.html file at the root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'test-embed.html'));
});

// Determine the host to listen on (0.0.0.0 for Docker, localhost for local dev)
// Check if we're in a production environment
const isProduction = process.env.NODE_ENV === 'production';
const host = '0.0.0.0'; // Always use 0.0.0.0 for cloud deployments

// Start the server
app.listen(port, host, () => {
  console.log(`Empress Naturals Chatbot dev server running at http://${host}:${port}`);
  console.log(`Server is listening on ${host}:${port}`);
  console.log(`Environment: ${isProduction ? 'Production' : 'Development'}`);
}); 