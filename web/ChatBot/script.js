document.addEventListener("DOMContentLoaded", function () {
    const chatLog = document.getElementById("chatLog");
    const userInput = document.getElementById("userInput");
    const sendButton = document.getElementById("sendButton");

    sendButton.addEventListener("click", function () {
        const userMessage = userInput.value;
        if (userMessage.trim() !== "") {
            displayMessage("user", userMessage);
            // Here, you can add code to process the user's message and provide a bot response
            // For example, simulate a bot response after a short delay:
            setTimeout(function () {
                const botResponse = "Thank you for your message. How can I assist you?";
                displayMessage("bot", botResponse);
            }, 500);
            userInput.value = "";
        }
    });

    function displayMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.className = sender;
        messageDiv.textContent = message;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }
});
