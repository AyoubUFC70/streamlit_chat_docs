css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e;
    color: #fff;
}
.chat-message.bot {
    background-color: #475063;
    color: #fff;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://e7.pngegg.com/pngimages/234/79/png-clipart-black-robot-face-illustration-robotics-technology-computer-icons-internet-bot-robotics-humanoid-robot-industrial-robot-thumbnail.png" style="max-height: 32px; max-width: 32px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://w7.pngwing.com/pngs/178/595/png-transparent-user-profile-computer-icons-login-user-avatars-thumbnail.png" style="max-height: 32px; max-width: 32px; border-radius: 50%; object-fit: cover;>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''