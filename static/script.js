// ===== Form Submission with Backend Integration =====
document.getElementById("uploadForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("fileInput");
  const conversionSelect = document.getElementById("conversion");
  const outputImage = document.getElementById("outputImage");
  const resultDiv = document.getElementById("result");
  const generateBtn = document.getElementById("generateBtn");

  if (!fileInput.files.length) return;

  const file = fileInput.files[0];
  const conversion = conversionSelect.value;

  generateBtn.disabled = true;
  generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';

  const formData = new FormData();
  formData.append("file", file);
  formData.append("conversion", conversion);

  try {
    const response = await fetch("/convert/", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.image) {
      outputImage.src = `data:image/png;base64,${data.image}`;
      outputImage.alt = `Face image after ${conversion === "young_to_old" ? "aging" : "de-aging"} transformation`;
      outputImage.style.display = "block";
      resultDiv.scrollIntoView({ behavior: "smooth" });
    } else {
      alert(data.error || "An error occurred.");
    }
  } catch (error) {
    console.error("Error:", error);
    alert("Something went wrong. Please try again later.");
  } finally {
    generateBtn.disabled = false;
    generateBtn.innerHTML = "Generate";
  }
});


// ===== Dark Mode Toggle =====
const darkToggle = document.getElementById("darkToggle");
const body = document.body;
const darkIcon = darkToggle.querySelector("i");

function setDarkMode(enabled) {
  if (enabled) {
    body.classList.add("dark");
    darkIcon.classList.replace("fa-moon", "fa-sun");
    darkToggle.setAttribute("aria-label", "Toggle light mode");
    darkToggle.setAttribute("title", "Toggle light mode");
  } else {
    body.classList.remove("dark");
    darkIcon.classList.replace("fa-sun", "fa-moon");
    darkToggle.setAttribute("aria-label", "Toggle dark mode");
    darkToggle.setAttribute("title", "Toggle dark mode");
  }
  localStorage.setItem("faceAgingDarkMode", enabled ? "true" : "false");
}




// ===== Mobile Menu Toggle =====
const mobileMenuButton = document.getElementById("mobileMenuButton");
const mobileMenu = document.getElementById("mobileMenu");

mobileMenuButton.addEventListener("click", () => {
  const expanded = mobileMenuButton.getAttribute("aria-expanded") === "true";
  mobileMenuButton.setAttribute("aria-expanded", !expanded);
  mobileMenu.classList.toggle("hidden");
});


// ===== Header Shadow on Scroll =====
const header = document.getElementById("header");
window.addEventListener("scroll", () => {
  if (window.scrollY > 10) {
    header.classList.add("scrolled");
  } else {
    header.classList.remove("scrolled");
  }
});


// ===== Scroll Animations =====
const scrollElements = document.querySelectorAll(".scroll-animate, .fade-in");
const scrollObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        scrollObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.15 }
);
scrollElements.forEach((el) => scrollObserver.observe(el));


// ===== Learn More Toggle =====
const learnMoreBtn = document.getElementById("learnMoreBtn");
const learnMoreContent = document.getElementById("learnMoreContent");

learnMoreBtn.addEventListener("click", () => {
  const isOpen = learnMoreContent.classList.toggle("open");
  learnMoreContent.hidden = !isOpen;
  learnMoreBtn.setAttribute("aria-expanded", isOpen);
  learnMoreBtn.textContent = isOpen ? "Show Less" : "Learn More";
});


// ===== Hero Upload Button Triggers File Picker =====
const uploadBtnHero = document.getElementById("uploadBtnHero");
const fileInput = document.getElementById("fileInput");

uploadBtnHero.addEventListener("click", () => {
  fileInput.click();
});
