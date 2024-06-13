use anyhow::{anyhow, Result};
use core::future::Future;
use futures::{executor, task};
use futures::{
    future::{BoxFuture, FutureExt},
    stream::Stream,
    task::{waker_ref, ArcWake},
    StreamExt,
};
use itertools::Itertools;
use log::{debug, info};
use reverse_search::guided_search::{search_wrapper, Executor, Guide};
use reverse_search::{ReverseSearchOut, Searcher, StepResult, TreeIndex};
use std::borrow::BorrowMut;
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::pin::Pin;
use std::process::Output;
use std::rc::Rc;
use std::task::{Poll, Wake, Waker};
use std::{
    pin::pin,
    sync::{mpsc, Arc, Mutex},
    thread,
};
use std::{thread_local, vec};
use task::Context;

thread_local! {
    static SEARCH: RefCell<Option<Searcher>> = RefCell::new(None);
}

struct FutureState {
    result: Option<Result<StepResult>>,
    waker: Option<Waker>,
}

struct StepFuture {
    state: Rc<RefCell<FutureState>>,
}

impl FutureState {
    fn set_result(&mut self, future_res: Result<StepResult>) {
        self.result.replace(future_res);
        if let Some(w) = self.waker.take() {
            w.wake();
        }
    }
}

impl Future for StepFuture {
    type Output = Result<StepResult>;

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut state = (*self.state).borrow_mut();
        if let Some(step_result) = state.result.take() {
            std::task::Poll::Ready(step_result)
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

pub struct GuideGenerator<'a> {
    pub guide: Guide,
    pub labels: &'a [usize],
}

impl GuideGenerator<'_> {
    fn generate<'b>(
        &'b self,
        executor: Rc<
            impl Fn(Vec<TreeIndex>) -> Pin<Box<dyn Future<Output = Result<StepResult>>>> + 'b,
        >,
    ) -> Option<Pin<Box<dyn Future<Output = Result<()>> + 'b>>> {
        let guided = self.guide.guided_search(executor);
        let future = async {
            let output = guided.await;
            match output {
                Ok(step_result) => {
                    if let Some(reverse_out) = step_result {
                        let matches: usize = reverse_out
                            .minkowski_decomp_iter()
                            .zip_eq(self.labels)
                            .map(|(a, b)| if a == *b { 1usize } else { 0usize })
                            .sum();
                        let accuracy = matches as f64 / self.labels.len() as f64;
                        info!(
                            "Got a guided result! {:?} with accuracy {}",
                            reverse_out.minkowski_decomp_iter().collect_vec(),
                            accuracy
                        );
                    } else {
                        debug!("No result 😔");
                    }
                    Ok(())
                }
                Err(err) => Err(err),
            }
        };
        let item = Box::pin(future);
        Some(item)
    }
}

pub struct RSWake {}

impl ArcWake for RSWake {
    fn wake_by_ref(_arc_self: &Arc<Self>) {
        debug!("Woken!")
    }
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<Rc<mpsc::Sender<Vec<TreeIndex>>>>,
    receiver: Option<mpsc::Receiver<(Vec<TreeIndex>, Result<StepResult>)>>,
}

impl ThreadPool {
    pub fn new(size: usize, search: Searcher) -> Self {
        assert!(size > 0);

        let (guide_sender, search_receiver) = mpsc::channel();
        let (search_sender, guide_receiver) = mpsc::channel();
        let search_receiver = Arc::new(Mutex::new(search_receiver));
        let search_sender = Arc::new(Mutex::new(search_sender));

        let mut workers = Vec::with_capacity(size);

        let search_ref = Arc::new(search);
        for id in 0..size {
            workers.push(Worker::new(
                id,
                Arc::clone(&search_receiver),
                Arc::clone(&search_sender),
                search_ref.clone(),
            ));
        }

        ThreadPool {
            workers,
            sender: Some(Rc::new(guide_sender)),
            receiver: Some(guide_receiver),
        }
    }

    pub fn run<'a>(&self, futures: &GuideGenerator, batch_size: usize) -> Result<()> {
        let wake: Arc<RSWake> = Arc::new(RSWake {});
        let waker = waker_ref(&wake);
        let context = &mut Context::from_waker(&waker);

        let mut tasks: Vec<Pin<Box<dyn Future<Output = Result<()>>>>> = Vec::new();
        let concurrent_tasks: Rc<Cell<HashMap<Vec<TreeIndex>, Rc<RefCell<FutureState>>>>> =
            Rc::new(Cell::new(HashMap::new()));
        let concurrent_tasks_ref = concurrent_tasks.clone();

        let sender = self.sender.as_ref().ok_or(anyhow!("No sender!"))?.clone();
        let executor = move |path: Vec<TreeIndex>| {
            sender.send(path.clone()).unwrap();
            let state = FutureState {
                result: None,
                waker: None,
            };
            let state_ref = Rc::new(RefCell::new(state));
            let mut concurrent_tasks = concurrent_tasks_ref.take();
            concurrent_tasks.insert(path, state_ref.clone());
            concurrent_tasks_ref.set(concurrent_tasks);
            let future: Pin<Box<dyn Future<Output = Result<StepResult>>>> =
                Box::pin(StepFuture { state: state_ref });
            future
        };
        let executor_ref = Rc::new(executor);

        loop {
            while tasks.len() < batch_size {
                if let Some(mut future) = futures.generate(executor_ref.clone()) {
                    match future.poll_unpin(context) {
                        Poll::Ready(_) => debug!("Poll ready"),
                        Poll::Pending => debug!("Poll pending"),
                    }
                    tasks.push(future);
                } else {
                    return Ok(());
                }
            }

            // We block here
            let receiver = self.receiver.as_ref().ok_or(anyhow!("No receiver"))?;
            // Received a message, unblock
            let (key, res) = receiver.recv()?;
            let concurrent_tasks_taken = concurrent_tasks.take();
            {
                let state = concurrent_tasks_taken
                    .get(&key)
                    .ok_or(anyhow!("Missing state key"))?;
                let mut mut_ref = (**state).try_borrow_mut()?;
                mut_ref.set_result(res);
            }
            concurrent_tasks.set(concurrent_tasks_taken);

            let filtered = tasks
                .drain(0..tasks.len())
                .filter_map(|mut task| match task.poll_unpin(context) {
                    Poll::Ready(_) => None,
                    Poll::Pending => Some(task),
                })
                .collect_vec();
            tasks.extend(filtered);
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.sender.take());
        drop(self.receiver.take());

        for worker in &mut self.workers {
            println!("Shutting down worker {}", worker.id);

            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
        receiver: Arc<Mutex<mpsc::Receiver<Vec<TreeIndex>>>>,
        sender: Arc<Mutex<mpsc::Sender<(Vec<TreeIndex>, Result<StepResult>)>>>,
        search: Arc<Searcher>,
    ) -> Worker {
        let thread = thread::spawn(move || loop {
            SEARCH.with_borrow_mut(|search_opt| {
                assert!(search_opt.is_none());
                search_opt.replace(search.as_ref().clone())
            });

            loop {
                let message = receiver.lock().unwrap().recv();

                let res = match message {
                    Ok(node_path) => {
                        debug!("Worker {id} got a job; executing.");
                        let res = SEARCH.with_borrow_mut(|search_opt| {
                            search_wrapper(search_opt.as_mut().unwrap(), node_path.clone())
                        });
                        sender.lock().unwrap().send((node_path, res))
                    }
                    Err(_) => {
                        info!("Worker {id} disconnected; shutting down.");
                        break;
                    }
                };
                if let Err(_) = res {
                    info!("Error sending on worker {id}; disconnecting");
                    break;
                }
            }
        });
        Worker {
            id,
            thread: Some(thread),
        }
    }
}
