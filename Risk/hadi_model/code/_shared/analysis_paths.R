# Shared paths and opt-in guards for long-running data/model stages.

source(file.path("code", "_shared", "paths.R"))

analysis_output_root <- Sys.getenv(
  "LDYG_ANALYSIS_OUTPUT_ROOT",
  unset = file.path(output_root, "analysis")
)

dir.create(analysis_output_root, recursive = TRUE, showWarnings = FALSE)

path_analysis_output <- function(...) file.path(analysis_output_root, ...)

allow_heavy <- function(flag) {
  identical(tolower(Sys.getenv(flag, unset = "false")), "true")
}

stop_unless_heavy_allowed <- function(flag, description) {
  if (!allow_heavy(flag)) {
    stop(
      description,
      "\nSet ", flag, "=true only when you intentionally want to run this long-running stage.",
      call. = FALSE
    )
  }
}
