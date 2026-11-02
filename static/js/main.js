/*
 * main.js
 * -------
 * Lightweight vanilla JavaScript for Smart Road Damage Detection.
 * Handles client-side file validation (JPG/PNG only, max 5 MB),
 * drag-and-drop dropzone interactions, instant image preview,
 * and loading spinner display during model inference.
 */

document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const dropzoneContent = document.getElementById("dropzone-content");
    const previewBox = document.getElementById("preview-box");
    const imagePreview = document.getElementById("image-preview");
    const previewFilename = document.getElementById("preview-filename");
    const previewFilesize = document.getElementById("preview-filesize");
    const removeFileBtn = document.getElementById("remove-file-btn");
    const uploadForm = document.getElementById("upload-form");
    const loadingOverlay = document.getElementById("loading-overlay");
    const submitBtn = document.getElementById("submit-btn");

    const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024; // 5 MB
    const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"];

    // Format bytes into human-readable size (e.g., 2.3 MB)
    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    }

    // Validate selected file type and size
    function validateFile(file) {
        if (!file) return false;

        // Check extension
        const ext = file.name.split(".").pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            alert("Invalid file format! Please upload a JPG or PNG road image.");
            return false;
        }

        // Check 5 MB limit
        if (file.size > MAX_FILE_SIZE_BYTES) {
            alert(`File too large! Your image is ${formatBytes(file.size)}, but the maximum limit is 5 MB.`);
            return false;
        }

        return true;
    }

    // Display image preview
    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewFilename.textContent = file.name;
            previewFilesize.textContent = "(" + formatBytes(file.size) + ")";
            dropzoneContent.style.display = "none";
            previewBox.style.display = "flex";
        };
        reader.readAsDataURL(file);
    }

    // Clear preview and reset file input
    function resetFileInput() {
        if (fileInput) fileInput.value = "";
        if (imagePreview) imagePreview.src = "#";
        if (dropzoneContent) dropzoneContent.style.display = "block";
        if (previewBox) previewBox.style.display = "none";
    }

    // File input change event
    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) {
                if (validateFile(file)) {
                    displayPreview(file);
                } else {
                    resetFileInput();
                }
            }
        });
    }

    // Drag and drop event listeners
    if (dropzone) {
        ["dragenter", "dragover"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                const file = files[0];
                if (validateFile(file)) {
                    // Update input files
                    fileInput.files = files;
                    displayPreview(file);
                }
            }
        });
    }

    // Remove file button
    if (removeFileBtn) {
        removeFileBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            resetFileInput();
        });
    }

    // Form submission: show loading spinner and disable button
    if (uploadForm) {
        uploadForm.addEventListener("submit", (e) => {
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                alert("Please select a road image file first!");
                e.preventDefault();
                return;
            }

            // Show loading modal
            if (loadingOverlay) {
                loadingOverlay.style.display = "flex";
            }

            // Disable submit button to prevent duplicate clicks
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.style.opacity = "0.7";
                submitBtn.style.cursor = "not-allowed";
            }
        });
    }
});
