<?php
header("Content-Type: text/plain");

$tableId = isset($_GET['table_id']) ? $_GET['table_id'] : '';

if (!empty($tableId)) {
    if (!preg_match('/^[a-zA-Z0-9_]+$/', $tableId)) {
        echo "ERR: invalid table_id";
        exit;
    }
    // Delete table via Rust API (handles: end game -> kill bots -> destroy)
    $url = "http://127.0.0.1:2230/api/v1/tables/" . urlencode($tableId);
    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "DELETE");
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 5);
    curl_exec($ch);
    curl_close($ch);
}

echo "OK";
?>