use anyhow::{anyhow, Result};
use core::future::Future;
use futures::task::WakerRef;
use futures::{executor, task};
use futures::{
    future::{BoxFuture, FutureExt},
    task::{waker_ref, ArcWake},
};
use itertools::Itertools;
use log::{debug, info};
use reverse_search::guided_search::{search_wrapper, Executor, Guide};
use reverse_search::{ReverseSearchOut, Searcher, StepResult, TreeIndex};
use serde_json::map::Iter;
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Poll, Wake, Waker};
use std::time;
use std::{
    pin::pin,
    sync::{mpsc, Arc, Mutex},
    thread,
};
use std::{thread_local, vec};
use task::Context;

const ten_secs: time::Duration = time::Duration::from_secs(10);

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

pub struct RSGen<'a> {
    tasks: Vec<Pin<Box<dyn Future<Output = Result<Option<ReverseSearchOut>>> + 'a>>>,
    concurrent_tasks: Rc<Cell<HashMap<Vec<TreeIndex>, Rc<RefCell<FutureState>>>>>,
    receiver: Rc<mpsc::Receiver<(Vec<TreeIndex>, Result<StepResult>)>>,
    wake: Arc<RSWake>,
    batch_size: usize,
    executor: Rc<
        Box<dyn Fn(Vec<TreeIndex>) -> Pin<Box<dyn Future<Output = Result<StepResult>> + 'a>> + 'a>,
    >,
    guide: &'a Guide,
}

impl<'a> RSGen<'a> {
    fn new(threadpool: &ThreadPool, guide: &'a Guide, batch_size: usize) -> Result<Self> {
        let wake: Arc<RSWake> = Arc::new(RSWake {});

        let tasks: Vec<Pin<Box<dyn Future<Output = Result<Option<ReverseSearchOut>>>>>> =
            Vec::new();
        let concurrent_tasks: Rc<Cell<HashMap<Vec<TreeIndex>, Rc<RefCell<FutureState>>>>> =
            Rc::new(Cell::new(HashMap::new()));
        let concurrent_tasks_ref = concurrent_tasks.clone();

        let sender: Rc<mpsc::Sender<Vec<TreeIndex>>> = threadpool
            .sender
            .as_ref()
            .ok_or(anyhow!("No sender!"))?
            .clone();
        let executor = move |path: Vec<TreeIndex>| {
            sender.send(path.clone()).unwrap();
            let state = FutureState {
                result: None,
                waker: None,
            };
            let state_ref = Rc::new(RefCell::new(state));
            let mut concurrent_tasks = concurrent_tasks_ref.take();
            let old_val = concurrent_tasks.insert(path, state_ref.clone());
            if let Some(_) = old_val {
                info!("Dupe key");
            }
            concurrent_tasks_ref.set(concurrent_tasks);
            let future: Pin<Box<dyn Future<Output = Result<StepResult>>>> =
                Box::pin(StepFuture { state: state_ref });
            future
        };
        let executor_ref: Rc<Box<dyn Fn(_) -> _ + 'a>> = Rc::new(Box::new(executor));

        let receiver = threadpool
            .receiver
            .as_ref()
            .ok_or(anyhow!("No receiver!"))?
            .clone();
        Ok(RSGen {
            wake,
            tasks,
            concurrent_tasks,
            receiver,
            batch_size,
            executor: executor_ref,
            guide,
        })
    }

    fn iterative_search(&mut self) -> Option<Result<Vec<ReverseSearchOut>>> {
        let waker = waker_ref(&self.wake);
        let context = &mut Context::from_waker(&waker);

        debug!("tasks len {}", self.tasks.len());
        while self.tasks.len() < self.batch_size {
            let mut future: Pin<Box<dyn Future<Output = Result<Option<ReverseSearchOut>>> + 'a>> =
                Box::pin(self.guide.guided_search(self.executor.clone()));
            match future.poll_unpin(context) {
                Poll::Ready(_) => debug!("Poll ready"),
                Poll::Pending => debug!("Poll pending"),
            }
            self.tasks.push(future);
        }


        let mut processed: Vec<
            std::result::Result<(Vec<TreeIndex>, Result<StepResult, _>), mpsc::RecvError>,
        > = Vec::new();
        debug!("Blocking whilst waiting for results");
        //thread::sleep(ten_secs);
        processed.push(self.receiver.recv()); // We block here
        loop {
            let received = self.receiver.try_recv();
            match received {
                Ok(msg) => processed.push(Ok(msg)),
                Err(err) => match err {
                    mpsc::TryRecvError::Empty => break,
                    mpsc::TryRecvError::Disconnected => {
                        processed.push(Err(mpsc::RecvError));
                        break;
                    }
                },
            }
        }

        let processed_transposed : Result<Vec<_>, _> = processed.into_iter().collect();
        match processed_transposed {
            Ok(step_results) => {
                debug!("Processed {}", step_results.len());
                let concurrent_tasks_taken = self.concurrent_tasks.take();
                for (key, step_result) in step_results {
                    let state_set_res = concurrent_tasks_taken
                    .get(&key)
                    .ok_or(anyhow!("Missing state key"))
                    .and_then(|state| {
                        (**state)
                            .try_borrow_mut()
                            .map_err(|err| anyhow!(err))
                    })
                    .and_then(|mut mut_ref| {
                        Ok(mut_ref.set_result(step_result))
                    });
                    if let Err(err) = state_set_res {
                        return Some(Err(anyhow!(err)));
                    }
                }
                self.concurrent_tasks.set(concurrent_tasks_taken);
            }
            Err(err) => return Some(Err(anyhow!(err))),
        }



        let (results, pending): (Vec<_>, Vec<_>) = self
            .tasks
            .drain(0..self.tasks.len())
            .map(|mut task| match task.poll_unpin(context) {
                Poll::Ready(out) => (Some(out), None),
                Poll::Pending => (None, Some(task)),
            })
            .unzip();
        self.tasks
            .extend(pending.into_iter().filter_map(|t| t).collect_vec());
        debug!("Still processing {}", self.tasks.len());
        let result_res: Result<Vec<_>> = results.into_iter().filter_map(|r| r).collect();
        Some(result_res.map(|res| res.into_iter().filter_map(|r| r).collect_vec()))
    }
}

impl Iterator for RSGen<'_> {
    type Item = Result<Vec<ReverseSearchOut>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterative_search()
    }
}

pub struct RSWake {

}

impl ArcWake for RSWake {
    fn wake_by_ref(_arc_self: &Arc<Self>) {
        debug!("Woken!")
    }
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<Rc<mpsc::Sender<Vec<TreeIndex>>>>,
    receiver: Option<Rc<mpsc::Receiver<(Vec<TreeIndex>, Result<StepResult>)>>>,
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
            receiver: Some(Rc::new(guide_receiver)),
        }
    }
}

pub fn runner<'a>(tp: &'a ThreadPool, guide: &'a Guide, batch_size: usize) -> Result<RSGen<'a>> {
    RSGen::new(&tp, guide, batch_size)
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.sender.take());
        drop(self.receiver.take());

        for worker in &mut self.workers {
            info!("Shutting down worker {}", worker.id);

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
        let thread = thread::spawn(move || {
            SEARCH.with_borrow_mut(|search_opt| {
                assert!(search_opt.is_none());
                search_opt.replace(search.as_ref().clone())
            });

            loop {
                //thread::sleep(hundred_millis);
                let message = receiver.lock().unwrap().recv();

                /*if id == 2{
                    info!("sleeping");
                    thread::sleep(ten_sec);
                }*/
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
