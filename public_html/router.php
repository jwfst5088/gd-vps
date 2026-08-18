<?php
$uri = $_SERVER['REQUEST_URI'];
$path = parse_url($uri, PHP_URL_PATH);

if (preg_match('#^/(api/|ping)#', $path)) {
    $backend = 'http://127.0.0.1:2230' . $uri;
    $ch = curl_init($backend);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HEADER, true);
    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
    curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $_SERVER['REQUEST_METHOD']);

    $headers = [];
    foreach (getallheaders() as $k => $v) {
        if (strtolower($k) !== 'host') {
            $headers[] = "$k: $v";
        }
    }
    if ($headers) curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

    $body = file_get_contents('php://input');
    if ($body) curl_setopt($ch, CURLOPT_POSTFIELDS, $body);

    $response = curl_exec($ch);
    $headerSize = curl_getinfo($ch, CURLINFO_HEADER_SIZE);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    $responseHeaders = substr($response, 0, $headerSize);
    $responseBody = substr($response, $headerSize);

    // Fix level swap bug (server assigns level to wrong team)
    if (strpos($uri, '/api/v1/tables') !== false && $httpCode === 200) {
        $data = json_decode($responseBody, true);
        if ($data) {
            $tables = isset($data['tables']) ? $data['tables'] : [$data];
            $isList = isset($data['tables']);

            foreach ($tables as &$table) {
                $state = isset($table['state']) ? $table['state'] : (isset($table['tableId']) ? $table : null);
                if (!$state || !isset($state['teams']) || count($state['teams']) !== 2) continue;
                if (!isset($state['scoreboard'])) continue;

                $teams = &$state['teams'];
                $sb = $state['scoreboard'];
                $winnerId = isset($sb['gameWinnerTeamId']) ? $sb['gameWinnerTeamId'] : null;
                if (!$winnerId) continue;

                $wIdx = ($teams[0]['teamId'] === $winnerId) ? 0 : 1;
                $lIdx = 1 - $wIdx;

                $levelOrder = ['2'=>2,'3'=>3,'4'=>4,'5'=>5,'6'=>6,'7'=>7,'8'=>8,'9'=>9,'10'=>10,'J'=>11,'Q'=>12,'K'=>13,'A'=>14];
                $wLevel = isset($teams[$wIdx]['level']) ? $teams[$wIdx]['level'] : '2';
                $lLevel = isset($teams[$lIdx]['level']) ? $teams[$lIdx]['level'] : '2';
                $wNum = isset($levelOrder[$wLevel]) ? $levelOrder[$wLevel] : intval($wLevel);
                $lNum = isset($levelOrder[$lLevel]) ? $levelOrder[$lLevel] : intval($lLevel);

                if ($lNum > $wNum && $wNum > 0) {
                    $tmp = $teams[$wIdx]['level'];
                    $teams[$wIdx]['level'] = $teams[$lIdx]['level'];
                    $teams[$lIdx]['level'] = $tmp;
                    $teams[$wIdx]['role'] = 'declarer';
                    $teams[$lIdx]['role'] = 'opponent';
                }
            }
            unset($table);

            $responseBody = json_encode($isList ? ['tables' => array_values($tables)] : (count($tables) > 0 ? array_values($tables)[0] : []));
        }
    }

    http_response_code($httpCode);
    foreach (explode("\r\n", $responseHeaders) as $h) {
        if (stripos($h, 'content-type:') !== false || stripos($h, 'x-') !== false) {
            header($h);
        }
    }
    echo $responseBody;
    exit;
}

return false;
