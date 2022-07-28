resource "aws_ecr_repository" "face_hugger_registry" {
  name                 = "face-hugger-registry"
  image_tag_mutability = "MUTABLE"
}
