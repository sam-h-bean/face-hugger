resource "aws_ecr_repository" "face_hugger_repository" {
  name                 = "face-hugger-repository"
  image_tag_mutability = "MUTABLE"
}
