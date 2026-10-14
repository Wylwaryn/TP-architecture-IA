# Lancement avec détection automatique du GPU (Windows PowerShell).
$gpu = $false
try {
    nvidia-smi -L > $null 2>&1
    if ($LASTEXITCODE -eq 0) { $gpu = $true }
} catch { }

if ($gpu) {
    Write-Host "GPU NVIDIA détecté -> Ollama sur GPU" -ForegroundColor Green
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
} else {
    Write-Host "Pas de GPU NVIDIA -> Ollama sur CPU" -ForegroundColor Yellow
    docker compose up --build
}
