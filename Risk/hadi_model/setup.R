# Setup checks for the replication package.
#
# Run this script from the package root before `source("main.R")`. By default it
# checks package availability and reports missing packages. It installs missing
# packages only when explicitly requested.

rootdir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

if (!file.exists(file.path(rootdir, "README.md")) ||
    !file.exists(file.path(rootdir, "env", "R_packages.txt"))) {
  stop(
    "setup.R must be run from the replication package root directory.",
    call. = FALSE
  )
}

required_r_version <- "4.5.2"
current_r_version <- paste(R.version$major, R.version$minor, sep = ".")

message("R version: ", current_r_version)
if (current_r_version != required_r_version) {
  message(
    "Note: this package was prepared with R ", required_r_version,
    "; results should be checked carefully if using another R version."
  )
}

package_file <- file.path(rootdir, "env", "R_packages.txt")
required_packages <- readLines(package_file, warn = FALSE)
required_packages <- trimws(required_packages)
required_packages <- required_packages[nzchar(required_packages)]
required_packages <- unique(required_packages)

installed <- rownames(utils::installed.packages())
missing_packages <- setdiff(required_packages, installed)

if (length(missing_packages) > 0) {
  message("Missing R packages:")
  message(paste(" -", missing_packages, collapse = "\n"))

  if (identical(Sys.getenv("LDYG_INSTALL_MISSING_PACKAGES"), "true")) {
    message("Installing missing packages because LDYG_INSTALL_MISSING_PACKAGES=true.")
    utils::install.packages(missing_packages)
  } else {
    stop(
      paste(
        "Install the missing packages listed above, or run:",
        "Sys.setenv(LDYG_INSTALL_MISSING_PACKAGES = 'true')",
        "source('setup.R')",
        "Alternatively, use renv::restore() with the included renv.lock file.",
        sep = "\n"
      ),
      call. = FALSE
    )
  }
}

missing_after_install <- setdiff(required_packages, rownames(utils::installed.packages()))
if (length(missing_after_install) > 0) {
  stop(
    "The following required packages are still missing after setup:\n",
    paste(missing_after_install, collapse = "\n"),
    call. = FALSE
  )
}

message("All packages listed in env/R_packages.txt are available.")

if (file.exists(file.path(rootdir, "renv.lock"))) {
  message("Optional renv.lock detected. Use renv::restore() for the recorded package versions if needed.")
}

message("Setup checks completed. You can now run source('main.R').")
