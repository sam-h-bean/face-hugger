terraform {
  cloud {
    organization = "data-conductors"

    workspaces {
      name = "face-hugger"
    }
  }
}
