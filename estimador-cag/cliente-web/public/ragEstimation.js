(function () {
  var modulesRoot = document.getElementById("modules");
  if (!modulesRoot) return;

  var nextModule = modulesRoot.querySelectorAll("[data-module]").length;
  var nextTask = 1000;

  function updateTotal() {
    var total = 0;
    modulesRoot.querySelectorAll("[data-days]").forEach(function (input) {
      var value = Number.parseInt(input.value, 10);
      if (!Number.isNaN(value)) total += value;
    });
    var target = document.querySelector("[data-total-days]");
    if (target) target.textContent = String(total);
  }

  function instantiate(templateId, replacements) {
    var template = document.getElementById(templateId);
    var html = template.innerHTML;
    Object.keys(replacements).forEach(function (key) {
      html = html.split(key).join(replacements[key]);
    });
    var wrapper = document.createElement("div");
    wrapper.innerHTML = html.trim();
    return wrapper.firstElementChild;
  }

  function addTask(moduleEl) {
    var task = instantiate("task-template", {
      __MODULE__: moduleEl.dataset.index,
      __TASK__: String(nextTask++),
    });
    moduleEl.querySelector("[data-tasks]").appendChild(task);
    updateTotal();
  }

  function addModule() {
    var index = String(nextModule++);
    var moduleEl = instantiate("module-template", { __MODULE__: index });
    moduleEl.dataset.index = index;
    modulesRoot.appendChild(moduleEl);
    updateTotal();
  }

  modulesRoot.querySelectorAll("[data-module]").forEach(function (moduleEl, index) {
    moduleEl.dataset.index = String(index);
  });

  document.addEventListener("click", function (event) {
    var button = event.target.closest("button");
    if (!button) return;
    if (button.id === "add-module") {
      addModule();
      return;
    }
    if (button.matches("[data-add-task]")) {
      addTask(button.closest("[data-module]"));
      return;
    }
    if (button.matches("[data-remove-task]")) {
      button.closest("[data-task]").remove();
      updateTotal();
      return;
    }
    if (button.matches("[data-remove-module]")) {
      button.closest("[data-module]").remove();
      updateTotal();
    }
  });

  modulesRoot.addEventListener("input", function (event) {
    if (event.target.matches("[data-days]")) updateTotal();
  });

  updateTotal();
})();
