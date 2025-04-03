(function() {
    // Configuration
    const config = {
        apiUrl: 'https://your-api-domain.com', // Replace with your actual API domain
        chatbotTitle: 'Express Analytics AI Assistant',
        logoUrl: 'https://www.expressanalytics.com/wp-content/uploads/2023/03/express-analytics-logo.svg',
        welcomeMessage: `👋 Welcome to Express Analytics! I'm your AI assistant, ready to help you with:
        <ul>
            <li>Data Analytics inquiries</li>
            <li>Marketing Analytics questions</li>
            <li>AI and Machine Learning solutions</li>
            <li>Business Intelligence insights</li>
        </ul>
        How can I assist you today?`,
        poweredByText: 'Powered by Express Analytics',
        poweredByLink: 'https://www.expressanalytics.com',
        position: 'right', // 'right' or 'left'
        initiallyOpen: false,
        buttonIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>'
    };

    // Create and inject CSS
    function injectStyles() {
        const style = document.createElement('style');
        style.textContent = `
            :root {
                --ea-primary-color: #0056b3;
                --ea-secondary-color: #f8f9fa;
                --ea-text-color: #333;
                --ea-light-gray: #e9ecef;
                --ea-border-color: #dee2e6;
                --ea-success-color: #28a745;
                --ea-user-message-bg: #e3f2fd;
                --ea-assistant-message-bg: #f8f9fa;
            }
            
            .ea-chat-widget {
                position: fixed;
                ${config.position}: 20px;
                bottom: 20px;
                z-index: 9999;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            }
            
            .ea-chat-button {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background-color: var(--ea-primary-color);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease;
            }
            
            .ea-chat-button:hover {
                transform: scale(1.05);
            }
            
            .ea-chat-container {
                position: absolute;
                bottom: 70px;
                ${config.position}: 0;
                width: 350px;
                height: 500px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 5px 25px rgba(0, 0, 0, 0.2);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                transition: all 0.3s ease;
                opacity: 0;
                transform: translateY(20px) scale(0.9);
                pointer-events: none;
            }
            
            .ea-chat-container.open {
                opacity: 1;
                transform: translateY(0) scale(1);
                pointer-events: all;
            }
            
            .ea-chat-header {
                background-color: white;
                padding: 15px;
                border-bottom: 1px solid var(--ea-border-color);
                display: flex;
                align-items: center;
            }
            
            .ea-logo {
                margin-right: 10px;
            }
            
            .ea-logo img {
                height: 24px;
            }
            
            .ea-chat-header h3 {
                font-size: 16px;
                font-weight: 600;
                color: var(--ea-primary-color);
                margin: 0;
            }
            
            .ea-close-button {
                margin-left: auto;
                background: none;
                border: none;
                color: #999;
                cursor: pointer;
                font-size: 20px;
                padding: 0;
                line-height: 1;
            }
            
            .ea-chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 15px;
            }
            
            .ea-message {
                margin-bottom: 15px;
                display: flex;
                flex-direction: column;
            }
            
            .ea-message.user {
                align-items: flex-end;
            }
            
            .ea-message.assistant {
                align-items: flex-start;
            }
            
            .ea-message-content {
                max-width: 85%;
                padding: 10px 12px;
                border-radius: 12px;
                font-size: 14px;
                line-height: 1.5;
            }
            
            .ea-user .ea-message-content {
                background-color: var(--ea-user-message-bg);
                color: #0c5460;
                border-top-right-radius: 4px;
            }
            
            .ea-assistant .ea-message-content {
                background-color: var(--ea-assistant-message-bg);
                color: var(--ea-text-color);
                border-top-left-radius: 4px;
            }
            
            .ea-message-content p {
                margin: 0 0 8px 0;
            }
            
            .ea-message-content p:last-child {
                margin-bottom: 0;
            }
            
            .ea-message-content ul {
                margin: 0 0 8px 0;
                padding-left: 20px;
            }
            
            .ea-chat-input-container {
                display: flex;
                padding: 10px;
                border-top: 1px solid var(--ea-border-color);
                background-color: white;
            }
            
            .ea-chat-input {
                flex: 1;
                border: 1px solid var(--ea-border-color);
                border-radius: 20px;
                padding: 8px 12px;
                font-size: 14px;
                resize: none;
                outline: none;
                font-family: inherit;
                max-height: 100px;
                overflow-y: auto;
            }
            
            .ea-chat-input:focus {
                border-color: var(--ea-primary-color);
            }
            
            .ea-send-button {
                background-color: var(--ea-primary-color);
                color: white;
                border: none;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                margin-left: 8px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background-color 0.2s;
            }
            
            .ea-send-button:hover {
                background-color: #004494;
            }
            
            .ea-send-button svg {
                width: 16px;
                height: 16px;
            }
            
            .ea-powered-by {
                text-align: center;
                font-size: 11px;
                color: #6c757d;
                padding: 5px;
                border-top: 1px solid var(--ea-border-color);
            }
            
            .ea-powered-by a {
                color: var(--ea-primary-color);
                text-decoration: none;
            }
            
            .ea-typing-indicator {
                display: flex;
                padding: 10px 12px;
                background-color: var(--ea-assistant-message-bg);
                border-radius: 12px;
                border-top-left-radius: 4px;
                max-width: 85%;
            }
            
            .ea-typing-indicator span {
                height: 7px;
                width: 7px;
                background-color: #bbb;
                border-radius: 50%;
                display: inline-block;
                margin-right: 4px;
                animation: ea-bounce 1.3s linear infinite;
            }
            
            .ea-typing-indicator span:nth-child(2) {
                animation-delay: 0.15s;
            }
            
            .ea-typing-indicator span:nth-child(3) {
                animation-delay: 0.3s;
                margin-right: 0;
            }
            
            @keyframes ea-bounce {
                0%, 60%, 100% {
                    transform: translateY(0);
                }
                30% {
                    transform: translateY(-4px);
                }
            }
            
            @media (max-width: 480px) {
                .ea-chat-container {
                    width: calc(100vw - 40px);
                    height: 60vh;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Create chat widget HTML
    function createChatWidget() {
        const widget = document.createElement('div');
        widget.className = 'ea-chat-widget';
        
        // Chat button
        const button = document.createElement('div');
        button.className = 'ea-chat-button';
        button.innerHTML = config.buttonIcon;
        button.setAttribute('aria-label', 'Open chat');
        button.setAttribute('role', 'button');
        button.setAttribute('tabindex', '0');
        
        // Chat container
        const container = document.createElement('div');
        container.className = 'ea-chat-container';
        if (config.initiallyOpen) {
            container.classList.add('open');
        }
        
        // Chat header
        const header = document.createElement('div');
        header.className = 'ea-chat-header';
        
        const logo = document.createElement('div');
        logo.className = 'ea-logo';
        logo.innerHTML = `<img src="${config.logoUrl}" alt="Logo">`;
        
        const title = document.createElement('h3');
        title.textContent = config.chatbotTitle;
        
        const closeButton = document.createElement('button');
        closeButton.className = 'ea-close-button';
        closeButton.innerHTML = '&times;';
        closeButton.setAttribute('aria-label', 'Close chat');
        
        header.appendChild(logo);
        header.appendChild(title);
        header.appendChild(closeButton);
        
        // Chat messages
        const messages = document.createElement('div');
        messages.className = 'ea-chat-messages';
        
        // Welcome message
        const welcomeMsg = document.createElement('div');
        welcomeMsg.className = 'ea-message ea-assistant';
        
        const welcomeContent = document.createElement('div');
        welcomeContent.className = 'ea-message-content';
        welcomeContent.innerHTML = config.welcomeMessage;
        
        welcomeMsg.appendChild(welcomeContent);
        messages.appendChild(welcomeMsg);
        
        // Chat input
        const inputContainer = document.createElement('div');
        inputContainer.className = 'ea-chat-input-container';
        
        const input = document.createElement('textarea');
        input.className = 'ea-chat-input';
        input.placeholder = 'Type your message here...';
        input.rows = 1;
        
        const sendButton = document.createElement('button');
        sendButton.className = 'ea-send-button';
        sendButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
        sendButton.setAttribute('aria-label', 'Send message');
        
        inputContainer.appendChild(input);
        inputContainer.appendChild(sendButton);
        
        // Powered by
        const poweredBy = document.createElement('div');
        poweredBy.className = 'ea-powered-by';
        poweredBy.innerHTML = `<a href="${config.poweredByLink}" target="_blank" rel="noopener noreferrer">${config.poweredByText}</a>`;
        
        // Assemble container
        container.appendChild(header);
        container.appendChild(messages);
        container.appendChild(inputContainer);
        container.appendChild(poweredBy);
        
        // Assemble widget
        widget.appendChild(button);
        widget.appendChild(container);
        
        return {
            widget,
            button,
            container,
            closeButton,
            messages,
            input,
            sendButton
        };
    }

    // Initialize chat functionality
    function initChat(elements) {
        const { button, container, closeButton, messages, input, sendButton } = elements;
        
        // Generate a unique session ID or retrieve from localStorage
        let sessionId = localStorage.getItem('ea_chat_session_id');
        if (!sessionId) {
            sessionId = generateSessionId();
            localStorage.setItem('ea_chat_session_id', sessionId);
        }
        
        // Toggle chat open/closed
        button.addEventListener('click', () => {
            container.classList.toggle('open');
            if (container.classList.contains('open')) {
                input.focus();
            }
        });
        
        // Close chat
        closeButton.addEventListener('click', () => {
            container.classList.remove('open');
        });
        
        // Auto-resize textarea as user types
        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
            
            // Reset height if empty
            if (this.value === '') {
                this.style.height = 'auto';
            }
        });
        
        // Send message when Enter key is pressed (without Shift)
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Send message when send button is clicked
        sendButton.addEventListener('click', sendMessage);
        
        // Function to send message
        function sendMessage() {
            const message = input.value.trim();
            if (message === '') return;
            
            // Add user message to chat
            addMessageToChat('user', message);
            
            // Clear input
            input.value = '';
            input.style.height = 'auto';
            
            // Show typing indicator
            showTypingIndicator();
            
            // Send message to API
            fetch(`${config.apiUrl}/chat`, {
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
                    throw new Error('Network response was not ok');
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
                addMessageToChat('assistant', 'Sorry, I encountered an error. Please try again later.');
            });
        }
        
        // Function to add message to chat
        function addMessageToChat(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `ea-message ea-${role}`;
            
            const messageContent = document.createElement('div');
            messageContent.className = 'ea-message-content';
            
            // Process markdown-like formatting
            content = processMarkdown(content);
            
            messageContent.innerHTML = content;
            messageDiv.appendChild(messageContent);
            
            messages.appendChild(messageDiv);
            
            // Scroll to bottom
            messages.scrollTop = messages.scrollHeight;
        }
        
        // Function to show typing indicator
        function showTypingIndicator() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'ea-message ea-assistant';
            typingDiv.id = 'ea-typing-indicator';
            
            const typingContent = document.createElement('div');
            typingContent.className = 'ea-typing-indicator';
            
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement('span');
                typingContent.appendChild(dot);
            }
            
            typingDiv.appendChild(typingContent);
            messages.appendChild(typingDiv);
            
            // Scroll to bottom
            messages.scrollTop = messages.scrollHeight;
        }
        
        // Function to remove typing indicator
        function removeTypingIndicator() {
            const typingIndicator = document.getElementById('ea-typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
        
        // Function to process markdown-like formatting
        function processMarkdown(text) {
            // Modified: Convert URLs to links, but only if they're not already inside an anchor tag
            // This regex uses negative lookahead and lookbehind to avoid URLs within existing href attributes
            text = text.replace(/(?<!href=["'])(?<!["'])(https?:\/\/[^\s"'<>]+)(?!["']>)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
            
            // Convert line breaks to <br>
            text = text.replace(/\n/g, '<br>');
            
            // Handle bullet points
            const bulletPointRegex = /^[-*•]\s+(.+)$/gm;
            text = text.replace(bulletPointRegex, '<li>$1</li>');
            
            // Wrap consecutive list items in <ul> tags
            text = text.replace(/(<li>.+<\/li>)\s*<br>(<li>.+<\/li>)/g, '$1$2');
            text = text.replace(/(<li>.+<\/li>)(?!\s*<li>)/g, '<ul>$1</ul>');
            
            // Handle bold text
            text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');
            
            // Handle italic text
            text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
            text = text.replace(/_(.*?)_/g, '<em>$1</em>');
            
            // Wrap paragraphs
            const paragraphs = text.split('<br><br>');
            text = paragraphs.map(p => {
                // Skip wrapping if it's a list or already wrapped
                if (p.startsWith('<ul>') || p.startsWith('<p>')) {
                    return p;
                }
                return `<p>${p}</p>`;
            }).join('');
            
            return text;
        }
        
        // Function to generate a session ID
        function generateSessionId() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }
        
        // Load chat history if available
        function loadChatHistory() {
            if (!sessionId) return;
            
            fetch(`${config.apiUrl}/history/${sessionId}`)
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
                    if (data && data.messages && data.messages.length > 0) {
                        // Clear default welcome message
                        messages.innerHTML = '';
                        
                        // Add messages to chat
                        data.messages.forEach(msg => {
                            addMessageToChat(msg.role, msg.content);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error loading chat history:', error);
                });
        }
        
        // Load chat history on initialization
        loadChatHistory();
    }

    // Initialize everything
    function init() {
        // Inject styles
        injectStyles();
        
        // Create chat widget
        const elements = createChatWidget();
        
        // Add to document
        document.body.appendChild(elements.widget);
        
        // Initialize chat functionality
        initChat(elements);
    }

    // Run initialization when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(); 