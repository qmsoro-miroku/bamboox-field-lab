
"use strict";

(() => {
  const root = document.documentElement;

  const toggle = document.getElementById("themeToggle");
  const themeLabel = document.getElementById("themeLabel");
  const statusRegion = document.getElementById("toastRegion");

  const media = window.matchMedia(
    "(prefers-color-scheme: dark)"
  );

  let preference = "system";

  function readPreference() {
    try {
      const stored = localStorage.getItem("bamboox-theme");

      return stored === "light" || stored === "dark"
        ? stored
        : "system";
    } catch (_) {
      return "system";
    }
  }

  function applyTheme() {
    const dark =
      preference === "dark" ||
      (preference === "system" && media.matches);

    root.dataset.theme = dark ? "dark" : "light";

    toggle?.setAttribute("aria-pressed", String(dark));

    toggle?.setAttribute(
      "aria-label",
      dark ? "Switch to light mode" : "Switch to dark mode"
    );

    if (themeLabel) {
      themeLabel.textContent = dark ? "Light mode" : "Dark mode";
    }

    document.dispatchEvent(
      new CustomEvent("bamboox:appearance", {
        detail: {
          preference,
          effectiveTheme: dark ? "dark" : "light"
        }
      })
    );

    window.dispatchEvent(new Event("bamboox:themechange"));
  }

  function setPreference(next) {
    if (!["light", "dark", "system"].includes(next)) {
      return;
    }

    preference = next;

    try {
      localStorage.setItem("bamboox-theme", next);
    } catch (_) {}

    applyTheme();
  }

  preference = readPreference();
  applyTheme();

  window.bambooxTheme = {
    getPreference: () => preference,
    setPreference
  };

  toggle?.addEventListener("click", () => {
    setPreference(
      root.dataset.theme === "dark" ? "light" : "dark"
    );
  });

  const onSystemThemeChange = () => {
    if (preference === "system") {
      applyTheme();
    }
  };

  if (media.addEventListener) {
    media.addEventListener("change", onSystemThemeChange);
  } else if (media.addListener) {
    media.addListener(onSystemThemeChange);
  }

  window.addEventListener("storage", event => {
    if (event.key === "bamboox-theme") {
      preference = readPreference();
      applyTheme();
    }
  });

  // Automatically show the active tab in the mobile topbar.
  document.querySelector(".fl-topnav .is-active")
    ?.scrollIntoView({
      block: "nearest",
      inline: "nearest"
    });

  // Preserve toast notifications used by existing pages.
  window.bambooxToast = (message, kind = "success") => {
    if (!statusRegion) return;

    const item = document.createElement("div");

    item.className =
      "toast " +
      (kind === "error" ? "toast-error" : "toast-success");

    item.setAttribute(
      "role",
      kind === "error" ? "alert" : "status"
    );

    const content = document.createElement("span");
    content.textContent = String(message);

    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.className = "toast-dismiss";
    dismiss.setAttribute(
      "aria-label",
      "Dismiss notification"
    );
    dismiss.textContent = "×";

    dismiss.addEventListener("click", () => item.remove());

    item.append(content, dismiss);
    statusRegion.append(item);

    window.setTimeout(
      () => item.remove(),
      kind === "error" ? 7000 : 4200
    );
  };
})();
