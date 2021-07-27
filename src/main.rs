use numpy::array::PyArrayDyn;
use numpy::Element;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{py_run, PyClass};
use std::collections::BTreeSet;

#[pyclass]
struct WriteOnce {
    inner: PyObject,
    record: BTreeSet<String>,
}

trait WriteOnceIndex {
    fn uget<T: Clone + PyClass>(&self, py: Python, attr: &str) -> PyResult<T>;
    fn uget_slice<T: Element>(&self, py: Python, attr: &str) -> PyResult<&[T]>;
    fn uget_slice_mut<T: Element>(&self, py: Python, attr: &str) -> PyResult<&mut [T]>;
}

#[pymethods]
impl WriteOnce {
    #[new]
    fn new(py: Python) -> Self {
        WriteOnce {
            inner: PyDict::new(py).into_py(py),
            record: Default::default(),
        }
    }
    fn add(&mut self, py: Python, attr: String, value: PyObject) -> PyResult<()> {
        if self.record.contains(&attr) {
            return Err(pyo3::exceptions::PyTypeError::new_err("repeat write"));
        }
        self.record.insert(attr.clone());
        self.inner.as_ref(py).set_item(attr, value)
    }

    fn get(&self, py: Python, attr: &str) -> PyResult<PyObject> {
        self.inner
            .as_ref(py)
            .get_item(attr)
            .map(|value| value.into_py(py))
    }
}

impl WriteOnceIndex for Py<WriteOnce> {
    fn uget<T: Clone + PyClass>(&self, py: Python, attr: &str) -> PyResult<T> {
        self.as_ref(py)
            .borrow()
            .inner
            .as_ref(py)
            .get_item(attr)?
            .extract()
    }
    fn uget_slice<T: Element>(&self, py: Python, attr: &str) -> PyResult<&[T]> {
        let m = self.as_ref(py).borrow();
        let array: &PyArrayDyn<T> = m.inner.as_ref(py).get_item(attr)?.extract()?;
        unsafe {
            let slice = array
                .as_slice()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err("not contiguous"))?;
            Ok(std::slice::from_raw_parts(slice.as_ptr(), slice.len()))
        }
    }
    fn uget_slice_mut<T: Element>(&self, py: Python, attr: &str) -> PyResult<&mut [T]> {
        let m = self.as_ref(py).borrow();
        let array: &PyArrayDyn<T> = m.inner.as_ref(py).get_item(attr)?.extract()?;
        unsafe {
            let slice = array
                .as_slice_mut()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err("not contiguous"))?;
            Ok(std::slice::from_raw_parts_mut(
                slice.as_mut_ptr(),
                slice.len(),
            ))
        }
    }
}

fn main() {
    pyo3::prepare_freethreaded_python();

    let write_once = Python::with_gil(|py| -> PyResult<_> {
        let write_once = Py::new(py, WriteOnce::new(py))?;
        py_run!(
            py,
            write_once,
            r#"
            import numpy as np
            write_once.add('list', np.absolute(np.array([-1, -2, -3], dtype='int32')))
            print('python:', write_once.get('list'))
        "#
        );
        Ok(write_once)
    })
    .unwrap();
    let list = Python::with_gil(|py| write_once.uget_slice::<i32>(py, "list"));
    println!("rust: {:?}", list);
}
