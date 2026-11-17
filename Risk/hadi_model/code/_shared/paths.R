# Shared paths for the manuscript analysis and figure scripts.

`%||%` <- function(x, y) if (is.null(x)) y else x

repo_root <- normalizePath(
  file.path(dirname(sys.frame(1)$ofile %||% getwd()), "..", ".."),
  winslash = "/",
  mustWork = FALSE
)

if (!dir.exists(file.path(repo_root, "code"))) {
  repo_root <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

data_root <- Sys.getenv(
  "LDYG_DATA_ROOT",
  unset = file.path(repo_root, "data")
)

output_root <- Sys.getenv(
  "LDYG_OUTPUT_ROOT",
  unset = file.path(repo_root, "output")
)

manuscript_output_dir <- output_root
manuscript_figure_dir <- file.path(manuscript_output_dir, "figures")
manuscript_figure_data_dir <- file.path(manuscript_output_dir, "figure_data")
manuscript_table_dir <- file.path(manuscript_output_dir, "tables")

dir.create(manuscript_figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(manuscript_figure_data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(manuscript_table_dir, recursive = TRUE, showWarnings = FALSE)

path_processed <- function(...) file.path(data_root, "processed", ...)
path_intermediate <- function(...) file.path(data_root, "intermediate", ...)
path_raw <- function(...) file.path(data_root, "raw", ...)
path_figure <- function(...) file.path(manuscript_figure_dir, ...)
path_figure_data <- function(...) file.path(manuscript_figure_data_dir, ...)
path_table <- function(...) file.path(manuscript_table_dir, ...)

assert_files_exist <- function(paths) {
  missing <- paths[!file.exists(paths)]
  if (length(missing) > 0) {
    stop("Missing required file(s):\n", paste(missing, collapse = "\n"), call. = FALSE)
  }
  invisible(paths)
}
