function copyCode(button) {
    // Find the parent code container of the clicked button
    const codeContainer = button.closest('.code-container');

    // Find the code block associated with the clicked button within the code container
    const codeBlock = codeContainer.querySelector('code');
    const text = codeBlock.textContent;

    // Create a textarea element to hold the code temporarily
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);

    // Select the text in the textarea
    textarea.select();
    textarea.setSelectionRange(0, 99999); // For mobile devices

    // Copy the text to the clipboard
    document.execCommand('copy');

    // Remove the textarea element
    document.body.removeChild(textarea);

    const img = button.querySelector('img');
    img.src = 'yes.png';

    // Hide the message and revert image after 2 seconds
    setTimeout(() => {
        //copiedMessage.style.display = 'none';
        img.src = "copycode.png";
    }, 3000);
}
