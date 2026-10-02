document.querySelectorAll("[data-form-loading]").forEach(function (form) {
  var upload = form.querySelector("[data-upload]");
  var textarea = form.querySelector("[data-textarea]");
  var status = form.querySelector("[data-upload-status]");
  var submit = form.querySelector("[data-submit]");
  var spinner = form.querySelector("[data-spinner]");
  var panel = form.querySelector("[data-status-panel]");

  if (upload && textarea) {
    upload.addEventListener("change", function () {
      var file = upload.files && upload.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function () {
        textarea.value = String(reader.result || "");
        if (status) status.textContent = file.name + " cargado";
      };
      reader.readAsText(file);
    });
  }

  form.addEventListener("submit", function () {
    if (submit) {
      submit.disabled = true;
      var label = submit.querySelector("[data-submit-label]");
      if (label) label.textContent = "Generando…";
    }
    if (spinner) spinner.classList.remove("hidden");
    if (panel) {
      panel.classList.remove("hidden");
      panel.classList.add("flex");
    }
  });
});
