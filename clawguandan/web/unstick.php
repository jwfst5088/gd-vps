<?php
header('Content-Type: application/json');

$tableId = $_GET['table_id'] ?? '';
if (!$tableId || !preg_match('/^[a-zA-Z0-9_]+$/', $tableId)) {
    echo json_encode(['ok' => false, 'error' => 'invalid table_id']);
    exit;
}

// Get current table state
$stateJson = file_get_contents("http://127.0.0.1:2230/api/v1/tables");
$state = json_decode($stateJson, true);

$found = false;
$phase = '';
$hand = null;
$stuckSeats = [];

foreach ($state['tables'] ?? [] as $t) {
    $st = $t['state'] ?? [];
    if (($st['tableId'] ?? '') === $tableId) {
        $found = true;
        $phase = $st['phase'] ?? '';
        $hand = $st['hand'] ?? null;
        $status = $st['status'] ?? '';
        $expect = $st['expect'] ?? [];
        $actors = $expect['actorPlayerIds'] ?? [];
        
        // Find stuck bot seats
        foreach ($st['seats'] ?? [] as $seat => $info) {
            if (($info['playerType'] ?? '') === 'bot' && in_array($info['playerId'] ?? '', $actors)) {
                $stuckSeats[] = ['seat' => $seat, 'playerId' => $info['playerId']];
            }
        }
        break;
    }
}

if (!$found) {
    echo json_encode(['ok' => false, 'error' => 'table not found']);
    exit;
}

// Check if actually stuck
$isStuck = ($status === 'in_game' && $phase === 'playing' && $hand === null && !empty($stuckSeats));
if (!$isStuck) {
    echo json_encode(['ok' => true, 'stuck' => false, 'phase' => $phase]);
    exit;
}

// Try to restart bots for stuck seats
$results = [];
foreach ($stuckSeats as $bot) {
    $ch = curl_init("http://127.0.0.1:2230/api/v1/tables/$tableId/bot/start");
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode([
        'playerName' => 'RuleBot',
        'timeoutMs' => 3000
    ]));
    curl_setopt($ch, CURLOPT_TIMEOUT, 5);
    $resp = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    $results[] = [
        'seat' => $bot['seat'],
        'httpCode' => $httpCode,
        'response' => json_decode($resp, true)
    ];
}

echo json_encode(['ok' => true, 'stuck' => true, 'results' => $results]);
