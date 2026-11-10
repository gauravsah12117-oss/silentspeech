// Screens viewer: the tabs swap the screenshot and its caption.
(() => {
  const tabs = document.querySelectorAll(".tabs button");
  const img = document.getElementById("screen-img");
  const cap = document.getElementById("screen-cap");
  tabs.forEach(tab => {
    new Image().src = tab.dataset.img;                       // preload, so switching is instant
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.setAttribute("aria-selected", String(t === tab)));
      img.src = tab.dataset.img;
      cap.textContent = tab.dataset.cap;
    });
  });
})();
