<?php
/**
 * Plugin Name: Express Analytics Chatbot
 * Plugin URI: https://www.expressanalytics.com
 * Description: AI-powered chatbot for Express Analytics that can be embedded on any WordPress site.
 * Version: 1.0.0
 * Author: Express Analytics
 * Author URI: https://www.expressanalytics.com
 * Text Domain: ea-chatbot
 */

// Exit if accessed directly
if (!defined('ABSPATH')) {
    exit;
}

// Define plugin constants
define('EA_CHATBOT_VERSION', '1.0.0');
define('EA_CHATBOT_PLUGIN_DIR', plugin_dir_path(__FILE__));
define('EA_CHATBOT_PLUGIN_URL', plugin_dir_url(__FILE__));

class EA_Chatbot {
    /**
     * Constructor
     */
    public function __construct() {
        // Register activation and deactivation hooks
        register_activation_hook(__FILE__, array($this, 'activate'));
        register_deactivation_hook(__FILE__, array($this, 'deactivate'));
        
        // Add settings page
        add_action('admin_menu', array($this, 'add_admin_menu'));
        add_action('admin_init', array($this, 'register_settings'));
        
        // Register shortcode
        add_shortcode('ea_chatbot', array($this, 'chatbot_shortcode'));
        
        // Enqueue scripts and styles
        add_action('wp_enqueue_scripts', array($this, 'enqueue_scripts'));
        
        // Add widget script to footer
        add_action('wp_footer', array($this, 'add_widget_script'));
    }
    
    /**
     * Plugin activation
     */
    public function activate() {
        // Set default options
        $default_options = array(
            'api_url' => 'https://your-api-domain.com',
            'chatbot_title' => 'Express Analytics AI Assistant',
            'logo_url' => 'https://www.expressanalytics.com/wp-content/uploads/2023/03/express-analytics-logo.svg',
            'welcome_message' => '👋 Welcome to Express Analytics! I\'m your AI assistant, ready to help you with:
            <ul>
                <li>Data Analytics inquiries</li>
                <li>Marketing Analytics questions</li>
                <li>AI and Machine Learning solutions</li>
                <li>Business Intelligence insights</li>
            </ul>
            How can I assist you today?',
            'powered_by_text' => 'Powered by Express Analytics',
            'powered_by_link' => 'https://www.expressanalytics.com',
            'position' => 'right',
            'initially_open' => false,
            'display_mode' => 'widget', // widget, inline, or disabled
            'primary_color' => '#0056b3'
        );
        
        add_option('ea_chatbot_options', $default_options);
    }
    
    /**
     * Plugin deactivation
     */
    public function deactivate() {
        // Cleanup if needed
    }
    
    /**
     * Add admin menu
     */
    public function add_admin_menu() {
        add_options_page(
            'Express Analytics Chatbot Settings',
            'EA Chatbot',
            'manage_options',
            'ea-chatbot',
            array($this, 'settings_page')
        );
    }
    
    /**
     * Register settings
     */
    public function register_settings() {
        register_setting('ea_chatbot_options_group', 'ea_chatbot_options');
        
        add_settings_section(
            'ea_chatbot_general_section',
            'General Settings',
            array($this, 'general_section_callback'),
            'ea-chatbot'
        );
        
        add_settings_field(
            'api_url',
            'API URL',
            array($this, 'api_url_callback'),
            'ea-chatbot',
            'ea_chatbot_general_section'
        );
        
        add_settings_field(
            'display_mode',
            'Display Mode',
            array($this, 'display_mode_callback'),
            'ea-chatbot',
            'ea_chatbot_general_section'
        );
        
        add_settings_field(
            'position',
            'Widget Position',
            array($this, 'position_callback'),
            'ea-chatbot',
            'ea_chatbot_general_section'
        );
        
        add_settings_field(
            'primary_color',
            'Primary Color',
            array($this, 'primary_color_callback'),
            'ea-chatbot',
            'ea_chatbot_general_section'
        );
        
        add_settings_field(
            'chatbot_title',
            'Chatbot Title',
            array($this, 'chatbot_title_callback'),
            'ea-chatbot',
            'ea_chatbot_general_section'
        );
        
        add_settings_field(
            'welcome_message',
            'Welcome Message',
            array($this, 'welcome_message_callback'),
            'ea-chatbot',
            'ea_chatbot_general_section'
        );
    }
    
    /**
     * General section callback
     */
    public function general_section_callback() {
        echo '<p>Configure your Express Analytics Chatbot settings below.</p>';
    }
    
    /**
     * API URL callback
     */
    public function api_url_callback() {
        $options = get_option('ea_chatbot_options');
        echo '<input type="text" id="api_url" name="ea_chatbot_options[api_url]" value="' . esc_attr($options['api_url']) . '" class="regular-text" />';
        echo '<p class="description">Enter the URL of your chatbot API (e.g., https://your-api-domain.com)</p>';
    }
    
    /**
     * Display mode callback
     */
    public function display_mode_callback() {
        $options = get_option('ea_chatbot_options');
        $display_mode = isset($options['display_mode']) ? $options['display_mode'] : 'widget';
        
        echo '<select id="display_mode" name="ea_chatbot_options[display_mode]">';
        echo '<option value="widget" ' . selected($display_mode, 'widget', false) . '>Floating Widget</option>';
        echo '<option value="inline" ' . selected($display_mode, 'inline', false) . '>Inline (via Shortcode)</option>';
        echo '<option value="disabled" ' . selected($display_mode, 'disabled', false) . '>Disabled</option>';
        echo '</select>';
        echo '<p class="description">Choose how the chatbot should be displayed on your site.</p>';
    }
    
    /**
     * Position callback
     */
    public function position_callback() {
        $options = get_option('ea_chatbot_options');
        $position = isset($options['position']) ? $options['position'] : 'right';
        
        echo '<select id="position" name="ea_chatbot_options[position]">';
        echo '<option value="right" ' . selected($position, 'right', false) . '>Right</option>';
        echo '<option value="left" ' . selected($position, 'left', false) . '>Left</option>';
        echo '</select>';
        echo '<p class="description">Choose the position of the floating chat widget.</p>';
    }
    
    /**
     * Primary color callback
     */
    public function primary_color_callback() {
        $options = get_option('ea_chatbot_options');
        $primary_color = isset($options['primary_color']) ? $options['primary_color'] : '#0056b3';
        
        echo '<input type="color" id="primary_color" name="ea_chatbot_options[primary_color]" value="' . esc_attr($primary_color) . '" />';
        echo '<p class="description">Choose the primary color for the chatbot.</p>';
    }
    
    /**
     * Chatbot title callback
     */
    public function chatbot_title_callback() {
        $options = get_option('ea_chatbot_options');
        echo '<input type="text" id="chatbot_title" name="ea_chatbot_options[chatbot_title]" value="' . esc_attr($options['chatbot_title']) . '" class="regular-text" />';
    }
    
    /**
     * Welcome message callback
     */
    public function welcome_message_callback() {
        $options = get_option('ea_chatbot_options');
        echo '<textarea id="welcome_message" name="ea_chatbot_options[welcome_message]" rows="6" class="large-text">' . esc_textarea($options['welcome_message']) . '</textarea>';
    }
    
    /**
     * Settings page
     */
    public function settings_page() {
        ?>
        <div class="wrap">
            <h1>Express Analytics Chatbot Settings</h1>
            <form method="post" action="options.php">
                <?php
                settings_fields('ea_chatbot_options_group');
                do_settings_sections('ea-chatbot');
                submit_button();
                ?>
            </form>
            
            <div class="card" style="max-width: 800px; margin-top: 20px; padding: 20px;">
                <h2>How to Use</h2>
                
                <h3>Option 1: Floating Widget</h3>
                <p>Select "Floating Widget" as the display mode to show the chatbot as a floating button on your site.</p>
                
                <h3>Option 2: Shortcode</h3>
                <p>Use the shortcode <code>[ea_chatbot]</code> to embed the chatbot inline on any page or post.</p>
                
                <h3>Option 3: Standalone Page</h3>
                <p>Create a standalone chatbot page by accessing:</p>
                <code><?php echo esc_url(plugins_url('static/index.html', __FILE__)); ?></code>
                
                <h3>Option 4: Embed via iframe</h3>
                <p>Embed the chatbot on any website using an iframe:</p>
                <code>&lt;iframe src="<?php echo esc_url(plugins_url('static/index.html', __FILE__)); ?>" width="100%" height="600px" frameborder="0"&gt;&lt;/iframe&gt;</code>
            </div>
        </div>
        <?php
    }
    
    /**
     * Enqueue scripts and styles
     */
    public function enqueue_scripts() {
        // Only enqueue for inline mode via shortcode
        if (is_admin()) {
            return;
        }
        
        $options = get_option('ea_chatbot_options');
        if (isset($options['display_mode']) && $options['display_mode'] === 'inline') {
            wp_enqueue_style('ea-chatbot-styles', plugins_url('static/styles.css', __FILE__), array(), EA_CHATBOT_VERSION);
            wp_enqueue_script('ea-chatbot-script', plugins_url('static/script.js', __FILE__), array(), EA_CHATBOT_VERSION, true);
            
            // Pass options to script
            wp_localize_script('ea-chatbot-script', 'eaChatbotOptions', array(
                'apiUrl' => $options['api_url'],
                'primaryColor' => $options['primary_color']
            ));
        }
    }
    
    /**
     * Add widget script to footer
     */
    public function add_widget_script() {
        $options = get_option('ea_chatbot_options');
        
        // Only add widget script if display mode is set to widget
        if (isset($options['display_mode']) && $options['display_mode'] === 'widget') {
            // Modify the configuration options
            $script = '<script>';
            $script .= 'window.eaChatbotConfig = {';
            $script .= 'apiUrl: "' . esc_js($options['api_url']) . '",';
            $script .= 'chatbotTitle: "' . esc_js($options['chatbot_title']) . '",';
            $script .= 'logoUrl: "' . esc_js($options['logo_url']) . '",';
            $script .= 'welcomeMessage: `' . esc_js($options['welcome_message']) . '`,';
            $script .= 'poweredByText: "' . esc_js($options['powered_by_text']) . '",';
            $script .= 'poweredByLink: "' . esc_js($options['powered_by_link']) . '",';
            $script .= 'position: "' . esc_js($options['position']) . '",';
            $script .= 'initiallyOpen: ' . ($options['initially_open'] ? 'true' : 'false') . ',';
            $script .= 'primaryColor: "' . esc_js($options['primary_color']) . '"';
            $script .= '};';
            $script .= '</script>';
            
            // Add the widget script
            $script .= '<script src="' . plugins_url('static/wordpress-embed.js', __FILE__) . '"></script>';
            
            echo $script;
        }
    }
    
    /**
     * Chatbot shortcode
     */
    public function chatbot_shortcode($atts) {
        $options = get_option('ea_chatbot_options');
        
        // If display mode is not inline, return empty
        if (!isset($options['display_mode']) || $options['display_mode'] !== 'inline') {
            return '';
        }
        
        // Parse attributes
        $atts = shortcode_atts(array(
            'height' => '500px',
            'width' => '100%'
        ), $atts);
        
        // Generate unique ID for this instance
        $chat_id = 'ea-chatbot-' . uniqid();
        
        // Start output buffering
        ob_start();
        ?>
        <div id="<?php echo esc_attr($chat_id); ?>" class="chat-container" style="height: <?php echo esc_attr($atts['height']); ?>; width: <?php echo esc_attr($atts['width']); ?>;">
            <div class="chat-header">
                <div class="logo">
                    <img src="<?php echo esc_url($options['logo_url']); ?>" alt="Express Analytics Logo">
                </div>
                <h1><?php echo esc_html($options['chatbot_title']); ?></h1>
            </div>
            <div class="chat-messages" id="<?php echo esc_attr($chat_id); ?>-messages">
                <div class="message assistant">
                    <div class="message-content">
                        <?php echo wp_kses_post($options['welcome_message']); ?>
                    </div>
                </div>
            </div>
            <div class="chat-input-container">
                <textarea id="<?php echo esc_attr($chat_id); ?>-input" placeholder="Type your message here..." rows="1"></textarea>
                <button id="<?php echo esc_attr($chat_id); ?>-send-button" aria-label="Send message">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
            <div class="powered-by">
                <a href="<?php echo esc_url($options['powered_by_link']); ?>" target="_blank" rel="noopener noreferrer"><?php echo esc_html($options['powered_by_text']); ?></a>
            </div>
        </div>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatId = '<?php echo esc_js($chat_id); ?>';
            const chatMessages = document.getElementById(chatId + '-messages');
            const chatInput = document.getElementById(chatId + '-input');
            const sendButton = document.getElementById(chatId + '-send-button');
            
            // API endpoint
            const API_URL = '<?php echo esc_js($options['api_url']); ?>/chat';
            
            // Generate a unique session ID or retrieve from localStorage
            let sessionId = localStorage.getItem('ea_chat_session_id');
            if (!sessionId) {
                sessionId = generateSessionId();
                localStorage.setItem('ea_chat_session_id', sessionId);
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
                fetch(API_URL, {
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
                typingDiv.id = chatId + '-typing-indicator';
                
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
                const typingIndicator = document.getElementById(chatId + '-typing-indicator');
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
                // Convert URLs to links
                text = text.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
                
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
            
            // Load chat history if available
            function loadChatHistory() {
                if (!sessionId) return;
                
                fetch(`<?php echo esc_js($options['api_url']); ?>/history/${sessionId}`)
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
                            chatMessages.innerHTML = '';
                            
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
            
            // Load chat history on page load
            loadChatHistory();
        });
        </script>
        <?php
        
        // Return the buffered content
        return ob_get_clean();
    }
}

// Initialize the plugin
$ea_chatbot = new EA_Chatbot(); 