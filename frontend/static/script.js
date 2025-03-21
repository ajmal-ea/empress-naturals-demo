document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    
    // Get API URL from config or use default
    const API_BASE_URL = window.EA_CHATBOT_CONFIG ? window.EA_CHATBOT_CONFIG.apiBaseUrl : 'https://empress-naturals-ea-bot-backend-production.up.railway.app';
    const API_CHAT_URL = `${API_BASE_URL}/chat`;
    
    // Generate a unique session ID or retrieve from localStorage
    let sessionId = localStorage.getItem('ea_chat_session_id');
    if (!sessionId) {
        sessionId = generateSessionId();
        localStorage.setItem('ea_chat_session_id', sessionId);
    }
    
    // Add reset button and contact us button to the header
    const chatHeader = document.querySelector('.chat-header');
    const headerActions = document.createElement('div');
    headerActions.className = 'header-actions';
    
    // Create reset button
    const resetButton = document.createElement('button');
    resetButton.id = 'reset-button';
    resetButton.title = 'Reset conversation';
    resetButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 2v6h6"></path><path d="M3 13a9 9 0 1 0 3-7.7L3 8"></path></svg>';
    resetButton.addEventListener('click', resetChat);
    
    // Create contact us button
    const contactButton = document.createElement('button');
    contactButton.id = 'contact-button';
    contactButton.title = 'Contact Us';
    contactButton.innerHTML = 'Contact Us';
    contactButton.addEventListener('click', toggleContactForm);
    
    headerActions.appendChild(resetButton);
    headerActions.appendChild(contactButton);
    chatHeader.appendChild(headerActions);
    
    // Create contact form container
    const contactFormContainer = document.createElement('div');
    contactFormContainer.id = 'contact-form-container';
    contactFormContainer.className = 'contact-form-container hidden';
    
    // Create contact form
    const contactForm = document.createElement('form');
    contactForm.id = 'hubspot-contact-form';
    contactForm.innerHTML = `
        <h3>Contact Us</h3>
        <div class="form-group">
            <label for="contact-name">Name*</label>
            <input type="text" id="contact-name" name="name" required>
        </div>
        <div class="form-group">
            <label for="contact-email">Company Email*</label>
            <input type="email" id="contact-email" name="email" required>
            <div class="error-message" id="email-error"></div>
        </div>
        <div class="form-group">
            <label for="contact-phone">Phone</label>
            <input type="tel" id="contact-phone" name="phone">
        </div>
        <div class="form-group">
            <label for="contact-company">Company Name</label>
            <input type="text" id="contact-company" name="company_name">
        </div>
        <div class="form-group">
            <label for="contact-note">Message</label>
            <textarea id="contact-note" name="note" rows="3"></textarea>
        </div>
        <div class="form-actions">
            <button type="button" id="cancel-contact">Cancel</button>
            <button type="submit" id="submit-contact" disabled>Submit</button>
        </div>
    `;
    
    // Insert the form before the chat input container
    const chatInputContainer = document.querySelector('.chat-input-container');
    document.querySelector('.chat-container').insertBefore(contactFormContainer, chatInputContainer);
    contactFormContainer.appendChild(contactForm);
    
    // Add event listener for form cancel button
    document.getElementById('cancel-contact').addEventListener('click', toggleContactForm);
    
    // Add event listener for form submission
    contactForm.addEventListener('submit', submitContactForm);
    
    // Function to toggle contact form visibility
    function toggleContactForm() {
        contactFormContainer.classList.toggle('hidden');
        // Reset form and error messages when toggling
        if (!contactFormContainer.classList.contains('hidden')) {
            contactForm.reset();
            document.getElementById('email-error').textContent = '';
        }
    }
    
    // Function to validate company email
    function isCompanyEmail(email) {
        // First check if email has valid format
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            return false;
        }
        
        // Check if email is not from common free email providers
        const freeEmailDomains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
            'aol.com', 'icloud.com', 'mail.com', 'protonmail.com',
            'zoho.com', 'yandex.com', 'gmx.com', 'live.com'
        ];
        
        const domain = email.split('@')[1].toLowerCase();
        return !freeEmailDomains.includes(domain);
    }
    
    // Function to submit contact form
    function submitContactForm(e) {
        e.preventDefault();
        
        // Get form data
        const name = document.getElementById('contact-name').value.trim();
        const email = document.getElementById('contact-email').value.trim();
        const phone = document.getElementById('contact-phone').value.trim();
        const company = document.getElementById('contact-company').value.trim();
        const note = document.getElementById('contact-note').value.trim();
        
        // Validate email is a company email
        if (!isCompanyEmail(email)) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                document.getElementById('email-error').textContent = 'Please enter a valid email address.';
            } else {
                document.getElementById('email-error').textContent = 'Please provide a company email address.';
            }
            return;
        }
        
        // Clear any previous error
        document.getElementById('email-error').textContent = '';
        
        // Disable submit button and show loading state
        const submitButton = document.getElementById('submit-contact');
        const originalButtonText = submitButton.textContent;
        submitButton.disabled = true;
        submitButton.textContent = 'Submitting...';
        
        // Send data to HubSpot endpoint
        fetch(`${API_BASE_URL}/create_hubspot_contact`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                email: email,
                phone: phone || null,
                company_name: company || null,
                note: note || null
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.detail || 'Failed to submit contact information');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide form
            toggleContactForm();
            
            // Add success message to chat
            addMessageToChat('assistant', `<p>Thank you for your interest! Your contact information has been submitted successfully. Our team will get in touch with you soon.</p>`);
            
            // Scroll to the success message
            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(error => {
            console.error('Error submitting contact form:', error);
            document.getElementById('email-error').textContent = error.message || 'An error occurred. Please try again.';
        })
        .finally(() => {
            // Re-enable submit button
            submitButton.disabled = false;
            submitButton.textContent = originalButtonText;
        });
    }
    
    // Auto-resize textarea as user types
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Reset height if empty
        if (this.value === '') {
            this.style.height = 'auto';
        }
    });
    
    // Send message when Enter key is pressed (without Shift)
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send message when send button is clicked
    sendButton.addEventListener('click', sendMessage);
    
    // Function to reset chat
    function resetChat() {
        // Clear chat messages
        chatMessages.innerHTML = '';
        
        // Generate new session ID
        sessionId = generateSessionId();
        localStorage.setItem('ea_chat_session_id', sessionId);
        
        // Add welcome message
        const welcomeMessage = window.EA_CHATBOT_CONFIG && window.EA_CHATBOT_CONFIG.welcomeMessage 
            ? window.EA_CHATBOT_CONFIG.welcomeMessage 
            : '<p>👋 Welcome to Empress Naturals! I\'m your skincare assistant, ready to help you with:</p><ul><li>Natural skincare for perimenopause and menopause</li><li>Information about our organic ingredients</li><li>Product recommendations for your skin concerns</li><li>Questions about our Empress Serums and body oils</li></ul><p>How can I assist with your royal skincare ritual today?</p>';
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = welcomeMessage;
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Briefly highlight the reset button to indicate success
        const resetButton = document.getElementById('reset-button');
        if (resetButton) {
            resetButton.classList.add('highlight-button');
            setTimeout(() => {
                resetButton.classList.remove('highlight-button');
            }, 1500);
        }
        
        // Hide contact form if it's open
        const contactFormContainer = document.getElementById('contact-form-container');
        if (contactFormContainer && !contactFormContainer.classList.contains('hidden')) {
            toggleContactForm();
        }
    }
    
    // Function to send message
    function sendMessage() {
        const message = chatInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send message to API
        fetch(API_CHAT_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId,
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.detail || 'Network response was not ok');
                });
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add bot response to chat
            addMessageToChat('assistant', data.response);
            
            // Update session ID if provided
            if (data.session_id) {
                sessionId = data.session_id;
                localStorage.setItem('ea_chat_session_id', sessionId);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            
            let errorMessage = 'Sorry, I encountered an error. Please try again later.';
            
            // Check for specific error messages
            if (error.message.includes('conversation has become too long')) {
                errorMessage = 'Your conversation has become too long. The system has attempted to summarize your conversation to maintain context while reducing size, but still encountered limits. Please try resetting the chat using the reset button in the top right corner.';
                
                // Add a visual indicator to the reset button to draw attention
                const resetButton = document.getElementById('reset-button');
                if (resetButton) {
                    resetButton.classList.add('highlight-button');
                    setTimeout(() => {
                        resetButton.classList.remove('highlight-button');
                    }, 3000);
                }
            } else if (error.message.includes('high demand')) {
                errorMessage = 'We\'re experiencing high demand right now. Please wait a moment and try again.';
            }
            
            addMessageToChat('assistant', errorMessage);
        });
    }
    
    // Function to add message to chat
    function addMessageToChat(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Process markdown-like formatting
        content = processMarkdown(content);
        
        messageContent.innerHTML = content;
        messageDiv.appendChild(messageContent);
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = 'typing-indicator';
        
        const typingContent = document.createElement('div');
        typingContent.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingContent.appendChild(dot);
        }
        
        typingDiv.appendChild(typingContent);
        chatMessages.appendChild(typingDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to remove typing indicator
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    // Function to generate a session ID
    function generateSessionId() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
    
    // Function to process markdown-like formatting
    function processMarkdown(text) {
        // Clean up the text first - ensure consistent line endings
        text = text.replace(/\r\n/g, '\n');
        
        // Handle headers (h1 through h4)
        text = text.replace(/#### (.*?)(\n|$)/g, '<h4>$1</h4>\n');
        text = text.replace(/### (.*?)(\n|$)/g, '<h3>$1</h3>\n');
        text = text.replace(/## (.*?)(\n|$)/g, '<h2>$1</h2>\n');
        text = text.replace(/# (.*?)(\n|$)/g, '<h1>$1</h1>\n');
        
        // Handle bold text (must be done before lists to avoid conflicts with asterisks)
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');
        
        // Convert URLs to links
        text = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
        
        // Split into paragraphs first (preserve empty lines)
        const paragraphs = text.split('\n\n');
        
        // Process each paragraph
        const processedParagraphs = paragraphs.map(paragraph => {
            // Skip if the paragraph is empty
            if (!paragraph.trim()) return '';
            
            // Check if this paragraph is a list
            if (/^\d+\.\s+/.test(paragraph)) {
                // Handle numbered lists
                const listItems = paragraph.split('\n');
                const processedItems = listItems.map(item => {
                    // Process numbered list items
                    return item.replace(/^\d+\.\s+(.+)$/, '<li>$1</li>');
                });
                return '<ol>' + processedItems.join('') + '</ol>';
            } 
            else if (/^[-*•]\s+/.test(paragraph)) {
                // Handle bullet points
                const listItems = paragraph.split('\n');
                const processedItems = listItems.map(item => {
                    // Process bullet points
                    return item.replace(/^[-*•]\s+(.+)$/, '<li>$1</li>');
                });
                return '<ul>' + processedItems.join('') + '</ul>';
            }
            else if (paragraph.startsWith('<h')) {
                // Already processed as a header
                return paragraph;
            }
            else {
                // Handle inline bullet points if any
                let processed = paragraph.replace(/([^\*])\s\*\s+([^*]+)$/gm, '$1<br><li>$2</li>');
                
                // Handle italic text (must be after lists)
                processed = processed.replace(/(?<!\<)\*([^\*]+)\*/g, '<em>$1</em>');
                processed = processed.replace(/(?<!\<)_([^_]+)_/g, '<em>$1</em>');
                
                // If we created list items, wrap them
                if (processed.includes('<li>')) {
                    processed = processed.replace(/(.+)(<li>.+<\/li>)/g, '<p>$1</p><ul>$2</ul>');
                } else {
                    // Regular paragraph
                    processed = '<p>' + processed + '</p>';
                }
                return processed;
            }
        });
        
        // Join all processed paragraphs
        return processedParagraphs.join('');
    }
    
    // Create history popup container
    const historyPopupContainer = document.createElement('div');
    historyPopupContainer.id = 'history-popup-container';
    historyPopupContainer.className = 'history-popup-container hidden';
    
    // Create history popup content
    const historyPopup = document.createElement('div');
    historyPopup.id = 'history-popup';
    historyPopup.innerHTML = `
        <div class="history-popup-header">
            <h3>Chat History</h3>
            <button id="close-history" aria-label="Close history">×</button>
        </div>
        <div class="history-popup-content" id="history-popup-content">
            <div class="history-loading">Loading history...</div>
        </div>
    `;
    
    // Add history popup to the document
    document.body.appendChild(historyPopupContainer);
    historyPopupContainer.appendChild(historyPopup);
    
    // Add event listener for close button
    document.getElementById('close-history').addEventListener('click', hideHistoryPopup);
    
    // Function to show history popup and load history
    function showHistoryPopup() {
        // Show the popup
        const historyPopupContainer = document.getElementById('history-popup-container');
        historyPopupContainer.classList.remove('hidden');
        
        // Show loading message
        const historyPopupContent = document.getElementById('history-popup-content');
        historyPopupContent.innerHTML = '<div class="history-loading">Loading history...</div>';
        
        // Fetch history data
        fetch(`${API_BASE_URL}/history/${sessionId}`)
            .then(response => {
                if (!response.ok) {
                    if (response.status === 404) {
                        // No history found
                        return null;
                    }
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data && data.messages && data.messages.length > 0) {
                    // Format the history data into conversation threads
                    const formattedHistory = formatHistoryData(data.messages);
                    historyPopupContent.innerHTML = formattedHistory;
                    
                    // Add event listeners to history items
                    addHistoryItemListeners();
                } else {
                    // No history found
                    historyPopupContent.innerHTML = '<div class="no-history">No chat history found for this session.</div>';
                }
            })
            .catch(error => {
                console.error('Error loading chat history:', error);
                historyPopupContent.innerHTML = '<div class="history-error">Error loading chat history. Please try again.</div>';
            });
    }
    
    // Function to hide history popup
    function hideHistoryPopup() {
        const historyPopupContainer = document.getElementById('history-popup-container');
        historyPopupContainer.classList.add('hidden');
    }
    
    // Function to format history data into conversation threads
    function formatHistoryData(messages) {
        if (!messages || messages.length === 0) {
            return '<div class="no-history">No chat history found.</div>';
        }
        
        // Group messages into conversation pairs (user + assistant)
        let html = '<div class="history-conversations">';
        
        for (let i = 0; i < messages.length; i += 2) {
            if (i + 1 < messages.length) {
                const userMessage = messages[i];
                const assistantMessage = messages[i + 1];
                
                if (userMessage.role === 'user' && assistantMessage.role === 'assistant') {
                    html += `
                        <div class="history-conversation">
                            <div class="history-message user">
                                <div class="history-message-header">You</div>
                                <div class="history-message-content">${userMessage.content}</div>
                            </div>
                            <div class="history-message assistant">
                                <div class="history-message-header">Assistant</div>
                                <div class="history-message-content">${assistantMessage.content}</div>
                            </div>
                        </div>
                    `;
                }
            }
        }
        
        html += '</div>';
        return html;
    }
    
    // Function to add event listeners to history items
    function addHistoryItemListeners() {
        // Add functionality to continue a conversation from history
        const historyConversations = document.querySelectorAll('.history-conversation');
        historyConversations.forEach(conversation => {
            conversation.addEventListener('click', () => {
                // Get the user query from this conversation
                const userQuery = conversation.querySelector('.history-message.user .history-message-content').innerHTML;
                
                // Close the history popup
                hideHistoryPopup();
                
                // Set the user query in the input field
                chatInput.value = userQuery.replace(/<[^>]*>/g, '');
                
                // Focus the input field
                chatInput.focus();
            });
        });
    }
    
    // Load chat history if available
    function loadChatHistory() {
        if (!sessionId) return;
        
        // Show loading indicator while fetching history
        showTypingIndicator();
        
        fetch(`${API_BASE_URL}/history/${sessionId}`)
            .then(response => {
                if (!response.ok) {
                    if (response.status === 404) {
                        // No history found, that's okay
                        return null;
                    }
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Remove typing indicator
                removeTypingIndicator();
                
                if (data && data.messages && data.messages.length > 0) {
                    // Clear default welcome message
                    chatMessages.innerHTML = '';
                    
                    // Add messages to chat
                    data.messages.forEach(msg => {
                        addMessageToChat(msg.role, msg.content);
                    });
                    
                    // Scroll to bottom
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                } else {
                    // If no history or empty history, show a message
                    if (chatMessages.children.length === 0) {
                        // Add welcome message if chat is empty
                        const welcomeMessage = window.EA_CHATBOT_CONFIG && window.EA_CHATBOT_CONFIG.welcomeMessage 
                            ? window.EA_CHATBOT_CONFIG.welcomeMessage 
                            : '<p>👋 Welcome to Empress Naturals! I\'m your skincare assistant, ready to help you with:</p><ul><li>Natural skincare for perimenopause and menopause</li><li>Information about our organic ingredients</li><li>Product recommendations for your skin concerns</li><li>Questions about our Empress Serums and body oils</li></ul><p>How can I assist with your royal skincare ritual today?</p>';
                        
                        addMessageToChat('assistant', welcomeMessage);
                    }
                }
            })
            .catch(error => {
                console.error('Error loading chat history:', error);
                removeTypingIndicator();
            });
    }
    
    // Load chat history on page load
    loadChatHistory();
}); 