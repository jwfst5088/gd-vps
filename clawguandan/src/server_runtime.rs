use clawguandan::api::{AppState, router};
use clawguandan::store::TableStore;
use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use tower_http::trace::TraceLayer;

pub async fn serve(ip: IpAddr, port: u16) -> Result<(), String> {
    clawguandan::learning::init_task_manager();
    clawguandan::learning::game_logger::init_global_logger(1);

    // 服务器重启后自动续传未完成的训练任务
    clawguandan::learning::auto_resume();

    std::thread::spawn(|| {
        loop {
            #[cfg(unix)]
            unsafe {
                let mut status: i32 = 0;
                loop {
                    let pid = libc::waitpid(-1, &mut status, libc::WNOHANG);
                    if pid <= 0 {
                        break;
                    }
                    eprintln!("INFO: zombie_reaper: reaped child pid={pid}");
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
    });

    let addr = SocketAddr::new(ip, port);
    let state = AppState {
        store: TableStore::new(),
        listen_port: port,
        bind_ip: ip,
        started_bot_tables: Arc::new(Mutex::new(HashSet::new())),
    };
    let app = router(state).layer(TraceLayer::new_for_http());

    tracing::info!(%addr, "clawguandan listening");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| e.to_string())?;
    axum::serve(listener, app).await.map_err(|e| e.to_string())
}