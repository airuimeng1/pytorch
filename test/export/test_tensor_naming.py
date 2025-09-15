# Owner(s): ["oncall: export"]

import torch
import torch.export
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTensorNamingForExport(TestCase):
    def test_basic_tensor_naming_api(self):
        """Test the basic tensor naming API."""
        # Test set_tensor_name
        tensor = torch.tensor([1, 2, 3])
        result = torch.export.set_tensor_name(tensor, "my_custom_tensor")
        self.assertIs(result, tensor)  # Should return the same tensor

        # Test name function (alias)
        tensor2 = torch.tensor([4, 5, 6])
        result2 = torch.export.name(tensor2, "my_name")
        self.assertIs(result2, tensor2)

    def test_tensor_naming_validation(self):
        """Test validation of tensor names."""
        tensor = torch.tensor([1, 2, 3])

        # Valid names should work
        torch.export.set_tensor_name(tensor, "valid_name")
        torch.export.set_tensor_name(tensor, "valid.name")
        torch.export.set_tensor_name(tensor, "valid123")

        # Invalid names should raise ValueError
        with self.assertRaises(ValueError):
            torch.export.set_tensor_name(tensor, "")  # Empty name

        with self.assertRaises(ValueError):
            torch.export.set_tensor_name(tensor, "invalid-name")  # Hyphen not allowed

        with self.assertRaises(ValueError):
            torch.export.set_tensor_name(tensor, "invalid name")  # Space not allowed

        # Invalid types should raise TypeError
        with self.assertRaises(TypeError):
            torch.export.set_tensor_name("not_a_tensor", "name")

        with self.assertRaises(TypeError):
            torch.export.set_tensor_name(tensor, 123)  # Name must be string

    def test_tensor_naming_with_export(self):
        """Test that custom tensor names are used during export."""

        # Create a simple module that uses a constant tensor
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Create a named tensor that will be lifted
                self.constant_tensor = torch.export.name(
                    torch.tensor([10, 20, 30]), "my_name_key"
                )

            def forward(self, x):
                return x + self.constant_tensor

        module = SimpleModule()
        example_input = torch.tensor([1, 2, 3])

        # Export the module
        exported_program = torch.export.export(module, (example_input,))

        # Check if our custom name appears in the graph signature
        lifted_tensor_names = list(
            exported_program.graph_signature.inputs_to_lifted_tensor_constants.values()
        )
        print(f"Lifted tensor names: {lifted_tensor_names}")

        # The custom name should be used (or a variation of it if there are conflicts)
        has_custom_name = any("my_name_key" in name for name in lifted_tensor_names)
        self.assertTrue(
            has_custom_name,
            f"Custom name 'my_name_key' not found in lifted tensor names: {lifted_tensor_names}",
        )

    def test_name_conflict_resolution(self):
        """Test that name conflicts are resolved properly."""

        class ModuleWithConflicts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Create two tensors with the same custom name
                self.tensor1 = torch.export.name(torch.tensor([1, 2]), "duplicate_name")
                self.tensor2 = torch.export.name(torch.tensor([3, 4]), "duplicate_name")

            def forward(self, x):
                return x + self.tensor1 + self.tensor2

        module = ModuleWithConflicts()
        example_input = torch.tensor([10, 20])

        # Export should handle name conflicts gracefully
        exported_program = torch.export.export(module, (example_input,))

        lifted_tensor_names = list(
            exported_program.graph_signature.inputs_to_lifted_tensor_constants.values()
        )
        print(f"Lifted tensor names with conflicts: {lifted_tensor_names}")

        # Should have two different names, both based on our custom name
        duplicate_names = [
            name for name in lifted_tensor_names if "duplicate_name" in name
        ]
        self.assertEqual(
            len(duplicate_names),
            2,
            f"Expected 2 names with 'duplicate_name', got: {duplicate_names}",
        )

        # Names should be different (one should have a suffix)
        self.assertNotEqual(duplicate_names[0], duplicate_names[1])

    def test_mixed_named_and_unnamed_tensors(self):
        """Test export with both named and unnamed tensors."""

        class MixedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Named tensor
                self.named_tensor = torch.export.name(
                    torch.tensor([100]), "custom_named"
                )
                # Unnamed tensor (should get module attribute name)
                self.unnamed_tensor = torch.tensor([200])

            def forward(self, x):
                return x + self.named_tensor + self.unnamed_tensor

        module = MixedModule()
        example_input = torch.tensor([1])

        exported_program = torch.export.export(module, (example_input,))

        lifted_tensor_names = list(
            exported_program.graph_signature.inputs_to_lifted_tensor_constants.values()
        )
        print(f"Mixed tensor names: {lifted_tensor_names}")

        # Should have both custom and module attribute names
        has_custom = any("custom_named" in name for name in lifted_tensor_names)
        has_module_attr = any("unnamed_tensor" in name for name in lifted_tensor_names)

        self.assertTrue(has_custom, f"Custom name not found in: {lifted_tensor_names}")
        self.assertTrue(
            has_module_attr,
            f"Module attribute name not found in: {lifted_tensor_names}",
        )

    def test_functional_correctness(self):
        """Test that naming doesn't affect the functional behavior."""

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.export.name(torch.tensor([5.0]), "my_constant")

            def forward(self, x):
                return x * self.const

        module = TestModule()
        input_tensor = torch.tensor([2.0])

        # Get original output
        original_output = module(input_tensor)

        # Export and run
        exported_program = torch.export.export(module, (input_tensor,))
        exported_output = exported_program.module()(input_tensor)

        # Results should be identical
        torch.testing.assert_close(original_output, exported_output)


if __name__ == "__main__":
    run_tests()
