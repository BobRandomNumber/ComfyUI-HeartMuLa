import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";

app.registerExtension({
    name: "HeartMuLa.FolderPicker",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "HeartMuLaLoader" || nodeData.name === "HeartMuLaCodecLoader" || nodeData.name === "HeartMuLaTranscriptionLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const pathWidget = this.widgets.find((w) => w.name === "base_path");
                
                this.addWidget("button", "📁 Select Base Path", null, () => {
                    showFolderPicker((selectedPath) => {
                        pathWidget.value = selectedPath;
                    });
                });

                return r;
            };
        }
    },
});

async function showFolderPicker(onSelect) {
    let currentPath = "";
    
    const dialog = $el("div", {
        style: {
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            backgroundColor: "#2a2a2a",
            color: "white",
            padding: "0",
            borderRadius: "8px",
            border: "1px solid #555",
            boxShadow: "0 10px 25px rgba(0,0,0,0.5)",
            zIndex: "10000",
            minWidth: "500px",
            width: "50vw",
            height: "60vh",
            display: "flex",
            flexDirection: "column",
            fontFamily: "Segoe UI, Tahoma, sans-serif"
        }
    });

    const header = $el("div", {
        style: {
            padding: "15px",
            borderBottom: "1px solid #444",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            backgroundColor: "#333",
            borderTopLeftRadius: "8px",
            borderTopRightRadius: "8px"
        }
    });
    header.appendChild($el("h3", { textContent: "Select Model Folder", style: { margin: "0", fontSize: "1.1em" } }));

    const content = $el("div", {
        style: {
            flex: "1",
            display: "flex",
            flexDirection: "column",
            padding: "15px",
            overflow: "hidden"
        }
    });

    const pathDisplay = $el("div", {
        style: {
            fontSize: "0.9em",
            marginBottom: "10px",
            padding: "8px",
            backgroundColor: "#1a1a1a",
            borderRadius: "4px",
            border: "1px solid #444",
            wordBreak: "break-all",
            color: "#6af"
        }
    });
    
    const listContainer = $el("div", {
        style: {
            flex: "1",
            overflowY: "auto",
            border: "1px solid #444",
            backgroundColor: "#111",
            borderRadius: "4px",
            padding: "0"
        }
    });

    const updateList = async (path) => {
        listContainer.innerHTML = "<div style='padding:20px; color:#888'>Loading...</div>";
        const response = await fetch(`/heartmula/browse?path=${encodeURIComponent(path)}`);
        const data = await response.json();
        
        if (data.error) {
            listContainer.innerHTML = `<div style='padding:20px; color:#f44'>Error: ${data.error}</div>`;
            return;
        }

        currentPath = data.current_path;
        pathDisplay.textContent = currentPath === "root" ? "Computer (Select Drive)" : currentPath;
        listContainer.innerHTML = "";

        const createItem = (label, pathValue, icon, isParent = false) => {
            const item = $el("div", {
                style: {
                    display: "flex",
                    alignItems: "center",
                    padding: "10px 15px",
                    cursor: "pointer",
                    borderBottom: "1px solid #222",
                    transition: "background 0.2s"
                },
                onmouseover: (e) => e.currentTarget.style.backgroundColor = "#333",
                onmouseout: (e) => e.currentTarget.style.backgroundColor = "transparent",
                onclick: () => updateList(pathValue)
            });
            
            const iconEl = $el("span", { textContent: icon, style: { marginRight: "10px", fontSize: "1.2em" } });
            const labelEl = $el("span", { textContent: label });
            
            item.appendChild(iconEl);
            item.appendChild(labelEl);
            return item;
        };

        // Add parent directory link if not at root
        if (data.parent) {
            listContainer.appendChild(createItem(".. [Up One Level]", data.parent, "⬆️", true));
        } else if (currentPath !== "root") {
             listContainer.appendChild(createItem(".. [To Computer]", "root", "💻", true));
        }

        data.dirs.forEach(dir => {
            const isDrive = dir.endsWith(':\\') || dir.endsWith(':/');
            const icon = isDrive ? "💽" : "📁";
            let fullPath = currentPath === "root" ? dir : (currentPath + "/" + dir);
            while (fullPath.includes("//")) fullPath = fullPath.replace("//", "/");
            listContainer.appendChild(createItem(dir, fullPath, icon));
        });
    };

    const footer = $el("div", { 
        style: { 
            padding: "15px", 
            borderTop: "1px solid #444", 
            display: "flex", 
            justifyContent: "flex-end", 
            gap: "10px",
            backgroundColor: "#333",
            borderBottomLeftRadius: "8px",
            borderBottomRightRadius: "8px"
        } 
    });
    
    const selectBtn = $el("button", {
        textContent: "Select Current Folder",
        style: { 
            padding: "8px 20px", 
            backgroundColor: "#4a4", 
            color: "white", 
            border: "none", 
            borderRadius: "4px",
            cursor: "pointer",
            fontWeight: "bold"
        },
        onclick: () => {
            if (currentPath === "root") return alert("Please select a drive first!");
            onSelect(currentPath);
            document.body.removeChild(dialog);
        }
    });

    const cancelBtn = $el("button", {
        textContent: "Cancel",
        style: { 
            padding: "8px 20px", 
            backgroundColor: "#555", 
            color: "white", 
            border: "none", 
            borderRadius: "4px",
            cursor: "pointer" 
        },
        onclick: () => document.body.removeChild(dialog)
    });

    footer.appendChild(cancelBtn);
    footer.appendChild(selectBtn);

    content.appendChild(pathDisplay);
    content.appendChild(listContainer);
    
    dialog.appendChild(header);
    dialog.appendChild(content);
    dialog.appendChild(footer);
    
    document.body.appendChild(dialog);
    updateList(""); // Start at default
}
