import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api/projects",
});

export const listProjects = () => API.get("/");

export const getProject = (id) => API.get(`/${id}/`);

export const createProject = ({ name, problem_description, dataset_file }) => {
  const form = new FormData();
  form.append("name", name);
  form.append("problem_description", problem_description || "");
  form.append("dataset_file", dataset_file);
  return API.post("/", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export default API;
