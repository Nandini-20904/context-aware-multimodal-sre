# 🔔 Notification System Configuration
# Email and Slack notification settings for SRE Incident Insight Engine

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EmailConfig:
    """Email notification configuration"""
    enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = ""
    to_emails: List[str] = None
    use_tls: bool = True
    
    def __post_init__(self):
        if self.to_emails is None:
            self.to_emails = []
        
        # Load from environment variables if available
        self.username = os.getenv("SMTP_USERNAME", self.username)
        self.password = os.getenv("SMTP_PASSWORD", self.password)
        self.from_email = os.getenv("FROM_EMAIL", self.from_email)

@dataclass
class SlackConfig:
    """Slack notification configuration"""
    enabled: bool = False
    webhook_url: str = ""
    channel: str = "#sre-alerts"
    username: str = "SRE-Engine"
    icon_emoji: str = ":robot_face:"
    mention_users: List[str] = None
    mention_channels: List[str] = None
    
    def __post_init__(self):
        if self.mention_users is None:
            self.mention_users = []
        if self.mention_channels is None:
            self.mention_channels = []
            
        # Load from environment variables if available
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", self.webhook_url)
        self.channel = os.getenv("SLACK_CHANNEL", self.channel)

@dataclass
class NotificationRules:
    """Notification filtering and routing rules"""
    notify_on_critical: bool = True
    notify_on_high: bool = True
    notify_on_medium: bool = False
    notify_on_low: bool = False
    notify_on_info: bool = False
    
    # Advanced rules
    business_hours_only: bool = False
    weekend_notifications: bool = True
    cooldown_minutes: int = 5
    max_notifications_per_hour: int = 20
    
    # Time-based rules
    business_start_hour: int = 9  # 9 AM
    business_end_hour: int = 18   # 6 PM
    timezone: str = "UTC"
    
    # Escalation rules
    escalation_enabled: bool = True
    escalation_time_minutes: int = 30
    escalation_emails: List[str] = None
    
    def __post_init__(self):
        if self.escalation_emails is None:
            self.escalation_emails = []

# Pre-configured notification templates
NOTIFICATION_TEMPLATES = {
    'critical_system_down': {
        'subject': '🚨 CRITICAL: System Component Down',
        'email_template': '''
        <h2 style="color: #d32f2f;">🚨 Critical System Alert</h2>
        <p><strong>Component:</strong> {component}</p>
        <p><strong>Issue:</strong> {message}</p>
        <p><strong>Impact:</strong> {impact}</p>
        <p><strong>Started:</strong> {timestamp}</p>
        <p><strong>Expected Resolution:</strong> {eta}</p>
        
        <h3>Immediate Actions Required:</h3>
        <ul>
            <li>Investigate root cause immediately</li>
            <li>Activate incident response team</li>
            <li>Implement emergency procedures</li>
            <li>Communicate with stakeholders</li>
        </ul>
        ''',
        'slack_template': '''
        :rotating_light: *CRITICAL SYSTEM ALERT* :rotating_light:
        
        *Component:* {component}
        *Issue:* {message}
        *Impact:* {impact}
        *Started:* {timestamp}
        *ETA:* {eta}
        
        *Immediate action required!*
        @here @channel
        '''
    },
    
    'high_performance_degradation': {
        'subject': '⚠️ HIGH: Performance Degradation Detected',
        'email_template': '''
        <h2 style="color: #f57c00;">⚠️ High Priority Alert</h2>
        <p><strong>Service:</strong> {component}</p>
        <p><strong>Issue:</strong> {message}</p>
        <p><strong>Metrics:</strong> {metrics}</p>
        <p><strong>Impact:</strong> {impact}</p>
        
        <h3>Recommended Actions:</h3>
        <ul>
            <li>Monitor trends closely</li>
            <li>Check resource utilization</li>
            <li>Review recent changes</li>
            <li>Consider scaling if needed</li>
        </ul>
        ''',
        'slack_template': '''
        :warning: *HIGH PRIORITY ALERT* :warning:
        
        *Service:* {component}
        *Issue:* {message}
        *Metrics:* {metrics}
        *Impact:* {impact}
        
        Please investigate and take appropriate action.
        '''
    },
    
    'security_incident': {
        'subject': '🛡️ SECURITY: Potential Security Incident',
        'email_template': '''
        <h2 style="color: #c62828;">🛡️ Security Alert</h2>
        <p><strong>Threat Type:</strong> {threat_type}</p>
        <p><strong>Description:</strong> {message}</p>
        <p><strong>Affected Systems:</strong> {component}</p>
        <p><strong>Confidence:</strong> {confidence}</p>
        
        <h3>Security Response:</h3>
        <ul>
            <li>Isolate affected systems if necessary</li>
            <li>Review access logs</li>
            <li>Contact security team</li>
            <li>Document all findings</li>
        </ul>
        ''',
        'slack_template': '''
        :shield: *SECURITY ALERT* :shield:
        
        *Threat:* {threat_type}
        *Description:* {message}
        *Systems:* {component}
        *Confidence:* {confidence}
        
        Security team please respond immediately.
        @security-team
        '''
    },
    
    'system_recovery': {
        'subject': '✅ RESOLVED: System Component Restored',
        'email_template': '''
        <h2 style="color: #388e3c;">✅ System Recovery</h2>
        <p><strong>Component:</strong> {component}</p>
        <p><strong>Resolution:</strong> {message}</p>
        <p><strong>Downtime:</strong> {duration}</p>
        <p><strong>Root Cause:</strong> {root_cause}</p>
        
        <h3>Recovery Summary:</h3>
        <ul>
            <li>Systems are now operational</li>
            <li>Performance metrics normalized</li>
            <li>Monitoring continues</li>
            <li>Post-incident review scheduled</li>
        </ul>
        ''',
        'slack_template': '''
        :white_check_mark: *SYSTEM RECOVERED* :white_check_mark:
        
        *Component:* {component}
        *Resolution:* {message}
        *Downtime:* {duration}
        *Root Cause:* {root_cause}
        
        All systems are now operational. Great work team!
        '''
    }
}

# Sample notification configurations for different environments
ENVIRONMENT_CONFIGS = {
    'production': {
        'email': EmailConfig(
            enabled=True,
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            to_emails=["sre-team@company.com", "oncall@company.com"]
        ),
        'slack': SlackConfig(
            enabled=True,
            channel="#production-alerts",
            mention_users=["@oncall", "@sre-lead"],
            mention_channels=["@channel"]
        ),
        'rules': NotificationRules(
            notify_on_critical=True,
            notify_on_high=True,
            notify_on_medium=True,
            cooldown_minutes=2,
            escalation_enabled=True,
            escalation_time_minutes=15
        )
    },
    
    'staging': {
        'email': EmailConfig(
            enabled=False  # Usually disabled for staging
        ),
        'slack': SlackConfig(
            enabled=True,
            channel="#staging-alerts",
            mention_users=[]  # No mentions for staging
        ),
        'rules': NotificationRules(
            notify_on_critical=True,
            notify_on_high=False,
            notify_on_medium=False,
            cooldown_minutes=10,
            escalation_enabled=False
        )
    },
    
    'development': {
        'email': EmailConfig(enabled=False),
        'slack': SlackConfig(
            enabled=True,
            channel="#dev-alerts",
            mention_users=[]
        ),
        'rules': NotificationRules(
            notify_on_critical=True,
            notify_on_high=False,
            notify_on_medium=False,
            cooldown_minutes=60,
            escalation_enabled=False
        )
    }
}

def get_config_for_environment(env: str = "production"):
    """Get notification configuration for specific environment"""
    return ENVIRONMENT_CONFIGS.get(env, ENVIRONMENT_CONFIGS['production'])

def validate_email_config(config: EmailConfig) -> List[str]:
    """Validate email configuration and return list of issues"""
    issues = []
    
    if config.enabled:
        if not config.smtp_server:
            issues.append("SMTP server is required")
        if not config.username:
            issues.append("SMTP username is required")
        if not config.password:
            issues.append("SMTP password is required")
        if not config.from_email:
            issues.append("From email address is required")
        if not config.to_emails:
            issues.append("At least one recipient email is required")
        
        # Validate email format (basic)
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if config.from_email and not re.match(email_pattern, config.from_email):
            issues.append("From email format is invalid")
        
        for email in config.to_emails:
            if not re.match(email_pattern, email):
                issues.append(f"Recipient email format is invalid: {email}")
    
    return issues

def validate_slack_config(config: SlackConfig) -> List[str]:
    """Validate Slack configuration and return list of issues"""
    issues = []
    
    if config.enabled:
        if not config.webhook_url:
            issues.append("Slack webhook URL is required")
        elif not config.webhook_url.startswith("https://hooks.slack.com"):
            issues.append("Slack webhook URL format is invalid")
        
        if not config.channel.startswith("#"):
            issues.append("Slack channel should start with #")
    
    return issues

# Environment variable loading helper
def load_from_environment():
    """Load configuration from environment variables"""
    return {
        'email': EmailConfig(
            enabled=os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower() == "true",
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            username=os.getenv("SMTP_USERNAME", ""),
            password=os.getenv("SMTP_PASSWORD", ""),
            from_email=os.getenv("FROM_EMAIL", ""),
            to_emails=os.getenv("TO_EMAILS", "").split(",") if os.getenv("TO_EMAILS") else []
        ),
        'slack': SlackConfig(
            enabled=os.getenv("SLACK_NOTIFICATIONS_ENABLED", "false").lower() == "true",
            webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
            channel=os.getenv("SLACK_CHANNEL", "#sre-alerts"),
            username=os.getenv("SLACK_USERNAME", "SRE-Engine")
        ),
        'rules': NotificationRules(
            notify_on_critical=os.getenv("NOTIFY_CRITICAL", "true").lower() == "true",
            notify_on_high=os.getenv("NOTIFY_HIGH", "true").lower() == "true",
            notify_on_medium=os.getenv("NOTIFY_MEDIUM", "false").lower() == "true",
            cooldown_minutes=int(os.getenv("NOTIFICATION_COOLDOWN", "5"))
        )
    }